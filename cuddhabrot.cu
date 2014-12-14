#include <chrono>
#include <cstdio>
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>

typedef unsigned long long int uint64;

using namespace std::chrono;

const uint64 SIDE_PX = 8000;

const double LEFT = -5.0 / 3;
const double TOP = -6.5 / 3;
const double SIDE = 10.0 / 3;
const double RIGHT = LEFT + SIDE;
const double BOTTOM = TOP + SIDE;

const int CELLS_PER_SIDE = 100;
const double CELL_SIZE = SIDE / CELLS_PER_SIDE;
const int TOTAL_CELLS = CELLS_PER_SIDE * CELLS_PER_SIDE;
const int ITERATIONS_PER_CELL = 20;
const int POINTS_PER_CELL = 10000;

const uint64 ITERATIONS_PER_POINT = 1000 * 1000;

const int THREADS = 48;
const int BLOCKS = 2560;
const int ITERATIONS_PER_BLOCK = 83;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

struct Complex
{
	__device__ Complex()
		: re(0.0), im(0.0) {}

	__device__ Complex(double real, double imag)
		: re(real), im(imag) {}

	__device__ double abs()
	{
		return re * re + im * im;
	}

	__device__ Complex operator+(Complex other)
	{
		return Complex(re + other.re, im + other.im); 
	}

	__device__ Complex operator-(Complex other)
	{
		return Complex(re - other.re, im - other.im);
	}

	__device__ Complex operator*(Complex other)
	{
		return Complex(re * other.re - im * other.im, re * other.im + im * other.re);
	}

	__device__ Complex operator/(Complex other)
	{
		double new_re = (re * other.re + im * other.im) / other.abs();
		double new_im = (im * other.re - re * other.im) / other.abs();
		return Complex(new_re, new_im);
	}

	__device__ bool operator==(Complex other)
	{
		return re == other.re && im == other.im;
	}

	__device__ Complex& operator=(Complex other)
	{
		re = other.re;
		im = other.im;
		return *this;
	}

	__device__ static Complex iterate(Complex x, Complex c)
	{
		double re = fma(x.re, x.re, c.re) - x.im * x.im;
		double im = fma(2*x.re, x.im, c.im);
		return Complex(re, im);
	}

	double re, im;
};

__global__ void init_rand(curandState_t* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(1337, idx, 0, &states[idx]);
}

__device__ Complex rand_complex_from_cell(curandState_t* rand_state, unsigned int cell_num)
{
	double cell_top = TOP + CELL_SIZE * (cell_num / CELLS_PER_SIDE);
	double cell_left = LEFT + CELL_SIZE * (cell_num % CELLS_PER_SIDE);
	double d1 = curand_uniform_double(rand_state);
	double d2 = curand_uniform_double(rand_state);
	double re = cell_top + CELL_SIZE * d1;
	double im = cell_left + CELL_SIZE * d2;
	return Complex(re, im);
}

__device__ bool outside(Complex x)
{
	return x.im < LEFT || x.im > RIGHT || x.re < TOP || x.re > BOTTOM;
}

struct Array
{
	unsigned int *data;
	size_t *size;
};

__global__ void find_edge_cells(curandState_t* rand_states, bool* on_edge)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= TOTAL_CELLS)
	{
		return;
	}
	bool has_converging = false;
	bool has_diverging = false;
	for (int point = 0; point < POINTS_PER_CELL && !(has_converging && has_diverging); point++)
	{
		Complex c, x;
		x = c = rand_complex_from_cell(&rand_states[idx], idx);
		int it;
		for (it = 0; it < ITERATIONS_PER_CELL; it++)
		{
			x = x * x + c;
			if (outside(x))
			{
				has_diverging = true;
				break;
			}
		}
		if (it == ITERATIONS_PER_CELL)
		{
			has_converging = true;
		}
	}
	on_edge[idx] = has_converging && has_diverging;
}

__device__ Complex rand_complex(curandState_t* rand_state, Array edge_cells)
{
	unsigned int cell = edge_cells.data[curand(rand_state) % *edge_cells.size];
	return rand_complex_from_cell(rand_state, cell);
}

// Add amount to circle of diameter 3 around pixel
__device__ void inc3(uint64* pic, Complex point, uint64 amount)
{
	size_t y = (point.re - TOP) / SIDE * SIDE_PX;
	size_t x = (point.im - LEFT) / SIDE * SIDE_PX;
	if (x >= SIDE_PX || y >= SIDE_PX)
	{
		return;
	}
	atomicAdd(&pic[y * SIDE_PX + x], amount);
	if (x > 0) atomicAdd(&pic[y * SIDE_PX + (x - 1)], amount);
	if (y > 0) atomicAdd(&pic[(y - 1) * SIDE_PX + x], amount);
	if (x < SIDE_PX - 1) atomicAdd(&pic[y * SIDE_PX + (x + 1)], amount);
	if (y < SIDE_PX - 1) atomicAdd(&pic[(y + 1) * SIDE_PX + x], amount);
}

__device__ bool is_power_of_two(uint64 x)
{
	return ((x != 0) && !(x & (x - 1)));
}

__global__ void generate(uint64* pic, curandState_t* rand_states, Array edge_cells)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	Complex init, c, x, old_x;
	curandState_t rand_state = rand_states[idx];

	for (int block_it = 0; block_it < ITERATIONS_PER_BLOCK; block_it++)
	{
		old_x = x = c = init = rand_complex(&rand_state, edge_cells);
		
		bool has_diverged = false;
		uint64 it;

		for (it = 0; it < ITERATIONS_PER_POINT; it++)
		{
			x = Complex::iterate(x, c);
			if (outside(x))
			{
				has_diverged = true;
				break;
			}
			if (x == old_x)
			{
				break;
			}
			if (is_power_of_two(it))
			{
				old_x = x;
			}
		}

		if (has_diverged)
		{
			x = c = init;
			inc3(pic, x, it);
			for (uint64 i = 0; i < it; i++)
			{
				x = Complex::iterate(x, c);
				inc3(pic, x, it);
			}
		}
	}
}

Array calc_edge_cells(curandState_t* rand_states)
{
	bool *on_edge;
	size_t on_edge_size = TOTAL_CELLS * sizeof(*on_edge);
	gpuErrchk(cudaMalloc(&on_edge, on_edge_size));
	find_edge_cells<<<BLOCKS, THREADS>>>(rand_states, on_edge);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	bool *on_edge_host;
	gpuErrchk(cudaMallocHost(&on_edge_host, on_edge_size));
	cudaMemcpy(on_edge_host, on_edge, on_edge_size, cudaMemcpyDeviceToHost);
	cudaFree(on_edge);

	size_t edge_count = 0;
	for (int i = 0; i < TOTAL_CELLS; i++)
	{
		if (on_edge_host[i])
		{
			edge_count++;
		}
	}

	unsigned int *edge_cells_host;
	size_t edge_cells_size = edge_count * sizeof(*edge_cells_host);
	gpuErrchk(cudaMallocHost(&edge_cells_host, edge_cells_size));
	
	edge_count = 0;
	for (int i = 0; i < TOTAL_CELLS; i++)
	{
		if (on_edge_host[i])
		{
			edge_cells_host[edge_count] = i;
			edge_count++;
		}
	}
	cudaFreeHost(on_edge_host);

	Array edge_cells;
	gpuErrchk(cudaMalloc(&edge_cells.data, edge_cells_size));
	gpuErrchk(cudaMalloc(&edge_cells.size, sizeof(edge_cells.size)));
	cudaMemcpy(edge_cells.data, edge_cells_host, edge_cells_size, cudaMemcpyHostToDevice);
	cudaMemcpy(edge_cells.size, &edge_count, sizeof(edge_cells.size), cudaMemcpyHostToDevice);

	cudaFreeHost(edge_cells_host);
	return edge_cells;
}

int main()
{
	gpuErrchk(cudaSetDevice(1));
	
	uint64 *pic;
	size_t pic_size = SIDE_PX * SIDE_PX * sizeof(*pic);
	gpuErrchk(cudaMalloc(&pic, pic_size));
	gpuErrchk(cudaMemset(pic, 0, pic_size));

	curandState_t *rand_states;
	size_t rand_states_size = BLOCKS * THREADS * sizeof(*rand_states);
	gpuErrchk(cudaMalloc(&rand_states, rand_states_size));

	init_rand<<<BLOCKS, THREADS>>>(rand_states);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	Array edge_cells = calc_edge_cells(rand_states);

	auto start = steady_clock::now();

	generate<<<BLOCKS, THREADS>>>(pic, rand_states, edge_cells);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	auto finish = steady_clock::now();
	auto elapsed_time = duration_cast<duration<double>>(finish - start);
	std::cout << "Elapsed time: " << elapsed_time.count() << "s" << std::endl;

	uint64 *host_pic;
	cudaMallocHost(&host_pic, pic_size);
	cudaMemcpy(host_pic, pic, pic_size, cudaMemcpyDeviceToHost);
	
	FILE *output = fopen("pic.bin", "wb");
	fwrite(host_pic, sizeof(*host_pic), SIDE_PX * SIDE_PX, output);
	fclose(output);

	cudaFreeHost(host_pic);
	cudaFree(edge_cells.data);
	cudaFree(edge_cells.size);
	cudaFree(rand_states);
	cudaFree(pic);
}
