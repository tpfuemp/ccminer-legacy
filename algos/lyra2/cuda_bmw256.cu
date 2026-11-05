#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_lyra2.h"

#undef SPH_ROTL32
#define SPH_ROTL32 ROTL32

__constant__ uint64_t pTarget[4];

#define shl(x, n) ((x) << (n))
#define shr(x, n) ((x) >> (n))

#define ss0(x) (shr((x), 1) ^ shl((x), 3) ^ SPH_ROTL32((x),  4) ^ SPH_ROTL32((x), 19))
#define ss1(x) (shr((x), 1) ^ shl((x), 2) ^ SPH_ROTL32((x),  8) ^ SPH_ROTL32((x), 23))
#define ss2(x) (shr((x), 2) ^ shl((x), 1) ^ SPH_ROTL32((x), 12) ^ SPH_ROTL32((x), 25))
#define ss3(x) (shr((x), 2) ^ shl((x), 2) ^ SPH_ROTL32((x), 15) ^ SPH_ROTL32((x), 29))
#define ss4(x) (shr((x), 1) ^ (x))
#define ss5(x) (shr((x), 2) ^ (x))

#define rs1(x) SPH_ROTL32((x),  3)
#define rs2(x) SPH_ROTL32((x),  7)
#define rs3(x) SPH_ROTL32((x), 13)
#define rs4(x) SPH_ROTL32((x), 16)
#define rs5(x) SPH_ROTL32((x), 19)
#define rs6(x) SPH_ROTL32((x), 23)
#define rs7(x) SPH_ROTL32((x), 27)

/* Message expansion function 1 */
__forceinline__ __device__
uint32_t expand32_1(int i, uint32_t *M32, const uint32_t *H, uint32_t *Q)
{
	return (ss1(Q[i - 16]) + ss2(Q[i - 15]) + ss3(Q[i - 14]) + ss0(Q[i - 13])
		+ ss1(Q[i - 12]) + ss2(Q[i - 11]) + ss3(Q[i - 10]) + ss0(Q[i - 9])
		+ ss1(Q[i - 8]) + ss2(Q[i - 7]) + ss3(Q[i - 6]) + ss0(Q[i - 5])
		+ ss1(Q[i - 4]) + ss2(Q[i - 3]) + ss3(Q[i - 2]) + ss0(Q[i - 1])
		+ ((i*(0x05555555ul) + SPH_ROTL32(M32[(i - 16) % 16], ((i - 16) % 16) + 1)
			+ SPH_ROTL32(M32[(i - 13) % 16], ((i - 13) % 16) + 1)
			- SPH_ROTL32(M32[(i - 6) % 16], ((i - 6) % 16) + 1)) ^ H[(i - 16 + 7) % 16]));
}

/* Message expansion function 2 */
__forceinline__ __device__
uint32_t expand32_2(int i, uint32_t *M32, const uint32_t *H, uint32_t *Q)
{
	return (Q[i - 16] + rs1(Q[i - 15]) + Q[i - 14] + rs2(Q[i - 13])
		+ Q[i - 12] + rs3(Q[i - 11]) + Q[i - 10] + rs4(Q[i - 9])
		+ Q[i - 8] + rs5(Q[i - 7]) + Q[i - 6] + rs6(Q[i - 5])
		+ Q[i - 4] + rs7(Q[i - 3]) + ss4(Q[i - 2]) + ss5(Q[i - 1])
		+ ((i*(0x05555555ul) + SPH_ROTL32(M32[(i - 16) % 16], ((i - 16) % 16) + 1)
			+ SPH_ROTL32(M32[(i - 13) % 16], ((i - 13) % 16) + 1)
			- SPH_ROTL32(M32[(i - 6) % 16], ((i - 6) % 16) + 1)) ^ H[(i - 16 + 7) % 16]));
}

__forceinline__ __device__
void Compression256(uint32_t *  M32)
{
	uint32_t Q[32], XL32, XH32;

	const uint32_t H[16] = {
		0x40414243, 0x44454647, 0x48494A4B, 0x4C4D4E4F,
		0x50515253, 0x54555657, 0x58595A5B, 0x5C5D5E5F,
		0x60616263, 0x64656667, 0x68696A6B, 0x6C6D6E6F,
		0x70717273, 0x74757677, 0x78797A7B, 0x7C7D7E7F
	};

	Q[0]  = (M32[5] ^ H[5]) - (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[13] ^ H[13]) + (M32[14] ^ H[14]);
	Q[1]  = (M32[6] ^ H[6]) - (M32[8] ^ H[8]) + (M32[11] ^ H[11]) + (M32[14] ^ H[14]) - (M32[15] ^ H[15]);
	Q[2]  = (M32[0] ^ H[0]) + (M32[7] ^ H[7]) + (M32[9]  ^ H[9])  - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[3]  = (M32[0] ^ H[0]) - (M32[1] ^ H[1]) + (M32[8]  ^ H[8])  - (M32[10] ^ H[10]) + (M32[13] ^ H[13]);
	Q[4]  = (M32[1] ^ H[1]) + (M32[2] ^ H[2]) + (M32[9]  ^ H[9])  - (M32[11] ^ H[11]) - (M32[14] ^ H[14]);
	Q[5]  = (M32[3] ^ H[3]) - (M32[2] ^ H[2]) + (M32[10] ^ H[10]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[6]  = (M32[4] ^ H[4]) - (M32[0] ^ H[0]) - (M32[3]  ^ H[3])  - (M32[11] ^ H[11]) + (M32[13] ^ H[13]);
	Q[7]  = (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[5]  ^ H[5])  - (M32[12] ^ H[12]) - (M32[14] ^ H[14]);
	Q[8]  = (M32[2] ^ H[2]) - (M32[5] ^ H[5]) - (M32[6]  ^ H[6])  + (M32[13] ^ H[13]) - (M32[15] ^ H[15]);
	Q[9]  = (M32[0] ^ H[0]) - (M32[3] ^ H[3]) + (M32[6]  ^ H[6])  - (M32[7]  ^ H[7])  + (M32[14] ^ H[14]);
	Q[10] = (M32[8] ^ H[8]) - (M32[1] ^ H[1]) - (M32[4]  ^ H[4])  - (M32[7]  ^ H[7])  + (M32[15] ^ H[15]);
	Q[11] = (M32[8] ^ H[8]) - (M32[0] ^ H[0]) - (M32[2]  ^ H[2])  - (M32[5]  ^ H[5])  + (M32[9]  ^ H[9]);
	Q[12] = (M32[1] ^ H[1]) + (M32[3] ^ H[3]) - (M32[6]  ^ H[6])  - (M32[9]  ^ H[9])  + (M32[10] ^ H[10]);
	Q[13] = (M32[2] ^ H[2]) + (M32[4] ^ H[4]) + (M32[7]  ^ H[7])  + (M32[10] ^ H[10]) + (M32[11] ^ H[11]);
	Q[14] = (M32[3] ^ H[3]) - (M32[5] ^ H[5]) + (M32[8]  ^ H[8])  - (M32[11] ^ H[11]) - (M32[12] ^ H[12]);
	Q[15] = (M32[12] ^ H[12]) - (M32[4] ^ H[4]) - (M32[6] ^ H[6]) - (M32[9]  ^ H[9])  + (M32[13] ^ H[13]);

	/* f_0: ssi, plus the previous double pipe */
	Q[0]  = ss0(Q[0])  + H[1];
	Q[1]  = ss1(Q[1])  + H[2];
	Q[2]  = ss2(Q[2])  + H[3];
	Q[3]  = ss3(Q[3])  + H[4];
	Q[4]  = ss4(Q[4])  + H[5];
	Q[5]  = ss0(Q[5])  + H[6];
	Q[6]  = ss1(Q[6])  + H[7];
	Q[7]  = ss2(Q[7])  + H[8];
	Q[8]  = ss3(Q[8])  + H[9];
	Q[9]  = ss4(Q[9])  + H[10];
	Q[10] = ss0(Q[10]) + H[11];
	Q[11] = ss1(Q[11]) + H[12];
	Q[12] = ss2(Q[12]) + H[13];
	Q[13] = ss3(Q[13]) + H[14];
	Q[14] = ss4(Q[14]) + H[15];
	Q[15] = ss0(Q[15]) + H[0];

	/* f_1: message expansion, 2 + 14 rounds */

	#pragma unroll
	for (int i=16; i<18; i++)
		Q[i] = expand32_1(i, M32, H, Q);

	#pragma nounroll
	for (int i=18; i<32; i++)
		Q[i] = expand32_2(i, M32, H, Q);

	/* XL, XH: XOR of the expanded words */
	XL32 = Q[16] ^ Q[17] ^ Q[18] ^ Q[19] ^ Q[20] ^ Q[21] ^ Q[22] ^ Q[23];
	XH32 = XL32^Q[24] ^ Q[25] ^ Q[26] ^ Q[27] ^ Q[28] ^ Q[29] ^ Q[30] ^ Q[31];


	/* f_2: double pipe for the next block */
	M32[0] = (shl(XH32, 5) ^ shr(Q[16], 5) ^ M32[0]) + (XL32    ^ Q[24] ^ Q[0]);
	M32[1] = (shr(XH32, 7) ^ shl(Q[17], 8) ^ M32[1]) + (XL32    ^ Q[25] ^ Q[1]);
	M32[2] = (shr(XH32, 5) ^ shl(Q[18], 5) ^ M32[2]) + (XL32    ^ Q[26] ^ Q[2]);
	M32[3] = (shr(XH32, 1) ^ shl(Q[19], 5) ^ M32[3]) + (XL32    ^ Q[27] ^ Q[3]);
	M32[4] = (shr(XH32, 3) ^ Q[20] ^ M32[4]) + (XL32    ^ Q[28] ^ Q[4]);
	M32[5] = (shl(XH32, 6) ^ shr(Q[21], 6) ^ M32[5]) + (XL32    ^ Q[29] ^ Q[5]);
	M32[6] = (shr(XH32, 4) ^ shl(Q[22], 6) ^ M32[6]) + (XL32    ^ Q[30] ^ Q[6]);
	M32[7] = (shr(XH32, 11) ^ shl(Q[23], 2) ^ M32[7]) + (XL32    ^ Q[31] ^ Q[7]);

	M32[8] = SPH_ROTL32(M32[4], 9) + (XH32     ^     Q[24] ^ M32[8]) + (shl(XL32, 8) ^ Q[23] ^ Q[8]);
	M32[9] = SPH_ROTL32(M32[5], 10) + (XH32     ^     Q[25] ^ M32[9]) + (shr(XL32, 6) ^ Q[16] ^ Q[9]);
	M32[10] = SPH_ROTL32(M32[6], 11) + (XH32     ^     Q[26] ^ M32[10]) + (shl(XL32, 6) ^ Q[17] ^ Q[10]);
	M32[11] = SPH_ROTL32(M32[7], 12) + (XH32     ^     Q[27] ^ M32[11]) + (shl(XL32, 4) ^ Q[18] ^ Q[11]);
	M32[12] = SPH_ROTL32(M32[0], 13) + (XH32     ^     Q[28] ^ M32[12]) + (shr(XL32, 3) ^ Q[19] ^ Q[12]);
	M32[13] = SPH_ROTL32(M32[1], 14) + (XH32     ^     Q[29] ^ M32[13]) + (shr(XL32, 4) ^ Q[20] ^ Q[13]);
	M32[14] = SPH_ROTL32(M32[2], 15) + (XH32     ^     Q[30] ^ M32[14]) + (shr(XL32, 7) ^ Q[21] ^ Q[14]);
	M32[15] = SPH_ROTL32(M32[3], 16) + (XH32     ^     Q[31] ^ M32[15]) + (shr(XL32, 2) ^ Q[22] ^ Q[15]);
}

__forceinline__ __device__
void Compression256_2(uint32_t *  M32)
{
	uint32_t XL32, XH32, Q[32];

	const uint32_t H[16] = {
		0xaaaaaaa0, 0xaaaaaaa1, 0xaaaaaaa2, 0xaaaaaaa3,
		0xaaaaaaa4, 0xaaaaaaa5, 0xaaaaaaa6, 0xaaaaaaa7,
		0xaaaaaaa8, 0xaaaaaaa9, 0xaaaaaaaa, 0xaaaaaaab,
		0xaaaaaaac, 0xaaaaaaad, 0xaaaaaaae, 0xaaaaaaaf
	};

	Q[0] = (M32[5] ^ H[5]) - (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[13] ^ H[13]) + (M32[14] ^ H[14]);
	Q[1] = (M32[6] ^ H[6]) - (M32[8] ^ H[8]) + (M32[11] ^ H[11]) + (M32[14] ^ H[14]) - (M32[15] ^ H[15]);
	Q[2] = (M32[0] ^ H[0]) + (M32[7] ^ H[7]) + (M32[9] ^ H[9]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[3] = (M32[0] ^ H[0]) - (M32[1] ^ H[1]) + (M32[8] ^ H[8]) - (M32[10] ^ H[10]) + (M32[13] ^ H[13]);
	Q[4] = (M32[1] ^ H[1]) + (M32[2] ^ H[2]) + (M32[9] ^ H[9]) - (M32[11] ^ H[11]) - (M32[14] ^ H[14]);
	Q[5] = (M32[3] ^ H[3]) - (M32[2] ^ H[2]) + (M32[10] ^ H[10]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[6] = (M32[4] ^ H[4]) - (M32[0] ^ H[0]) - (M32[3] ^ H[3]) - (M32[11] ^ H[11]) + (M32[13] ^ H[13]);
	Q[7] = (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[5] ^ H[5]) - (M32[12] ^ H[12]) - (M32[14] ^ H[14]);
	Q[8] = (M32[2] ^ H[2]) - (M32[5] ^ H[5]) - (M32[6] ^ H[6]) + (M32[13] ^ H[13]) - (M32[15] ^ H[15]);
	Q[9] = (M32[0] ^ H[0]) - (M32[3] ^ H[3]) + (M32[6] ^ H[6]) - (M32[7] ^ H[7]) + (M32[14] ^ H[14]);
	Q[10] = (M32[8] ^ H[8]) - (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[7] ^ H[7]) + (M32[15] ^ H[15]);
	Q[11] = (M32[8] ^ H[8]) - (M32[0] ^ H[0]) - (M32[2] ^ H[2]) - (M32[5] ^ H[5]) + (M32[9] ^ H[9]);
	Q[12] = (M32[1] ^ H[1]) + (M32[3] ^ H[3]) - (M32[6] ^ H[6]) - (M32[9] ^ H[9]) + (M32[10] ^ H[10]);
	Q[13] = (M32[2] ^ H[2]) + (M32[4] ^ H[4]) + (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[11] ^ H[11]);
	Q[14] = (M32[3] ^ H[3]) - (M32[5] ^ H[5]) + (M32[8] ^ H[8]) - (M32[11] ^ H[11]) - (M32[12] ^ H[12]);
	Q[15] = (M32[12] ^ H[12]) - (M32[4] ^ H[4]) - (M32[6] ^ H[6]) - (M32[9] ^ H[9]) + (M32[13] ^ H[13]);

	/* f_0: ssi, plus the previous double pipe */
	Q[0] = ss0(Q[0]) + H[1];
	Q[1] = ss1(Q[1]) + H[2];
	Q[2] = ss2(Q[2]) + H[3];
	Q[3] = ss3(Q[3]) + H[4];
	Q[4] = ss4(Q[4]) + H[5];
	Q[5] = ss0(Q[5]) + H[6];
	Q[6] = ss1(Q[6]) + H[7];
	Q[7] = ss2(Q[7]) + H[8];
	Q[8] = ss3(Q[8]) + H[9];
	Q[9] = ss4(Q[9]) + H[10];
	Q[10] = ss0(Q[10]) + H[11];
	Q[11] = ss1(Q[11]) + H[12];
	Q[12] = ss2(Q[12]) + H[13];
	Q[13] = ss3(Q[13]) + H[14];
	Q[14] = ss4(Q[14]) + H[15];
	Q[15] = ss0(Q[15]) + H[0];

	/* f_1: message expansion, 2 + 14 rounds */

	#pragma unroll
	for (int i = 16; i<18; i++)
		Q[i] = expand32_1(i, M32, H, Q);

	#pragma nounroll
	for (int i = 18; i<32; i++)
		Q[i] = expand32_2(i, M32, H, Q);

	/* XL, XH: XOR of the expanded words */
	XL32 = Q[16] ^ Q[17] ^ Q[18] ^ Q[19] ^ Q[20] ^ Q[21] ^ Q[22] ^ Q[23];
	XH32 = XL32 ^ Q[24] ^ Q[25] ^ Q[26] ^ Q[27] ^ Q[28] ^ Q[29] ^ Q[30] ^ Q[31];

	M32[2] = (shr(XH32, 5) ^ shl(Q[18], 5) ^ M32[2]) + (XL32 ^ Q[26] ^ Q[2]);
	M32[3] = (shr(XH32, 1) ^ shl(Q[19], 5) ^ M32[3]) + (XL32 ^ Q[27] ^ Q[3]);
	M32[14] = SPH_ROTL32(M32[2], 15) + (XH32 ^ Q[30] ^ M32[14]) + (shr(XL32, 7) ^ Q[21] ^ Q[14]);
	M32[15] = SPH_ROTL32(M32[3], 16) + (XH32 ^ Q[31] ^ M32[15]) + (shr(XL32, 2) ^ Q[22] ^ Q[15]);
}

#define TPB 512
__global__ __launch_bounds__(TPB, 2)
void bmw256_gpu_hash_32(uint32_t threads, uint32_t count, uint32_t startNounce, uint64_t *g_hash, uint32_t *resNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t message[16] = { 0 };

		LOHI(message[0], message[1], __ldg(&g_hash[thread]));
		LOHI(message[2], message[3], __ldg(&g_hash[thread + 1 * threads]));
		LOHI(message[4], message[5], __ldg(&g_hash[thread + 2 * threads]));
		LOHI(message[6], message[7], __ldg(&g_hash[thread + 3 * threads]));

		message[8]=0x80;
		message[14]=0x100;
		Compression256(message);
		Compression256_2(message);

		// hash words 6..7 against the target top; padding lanes past count never report
		if (thread < count && ((uint64_t*)message)[7] <= pTarget[3]) {
			const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
			if (pos < BMW256_MAX_RESULTS)
				resNonce[pos] = startNounce + thread;
		}
	}
}

__host__
void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t count, uint32_t startNounce, uint64_t *g_hash, uint32_t *d_resNonce)
{
	dim3 grid((threads + TPB - 1) / TPB);
	dim3 block(TPB);

	cudaMemset(d_resNonce, 0, sizeof(uint32_t));
	bmw256_gpu_hash_32 <<<grid, block>>> (threads, count, startNounce, g_hash, d_resNonce);
}

__host__
void bmw256_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32, 0, cudaMemcpyHostToDevice);
}
