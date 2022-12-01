#include <stdio.h>

texture<float, 1, cudaReadModeElementType> TableTex;
texture<float, 1, cudaReadModeElementType> PotTex;

struct col
{
	float x;
	float y;
	float z;
	float w;
};

#define BLOCK_SIZE 16
#define GRID_SIZE 256

__global__ void
kernel(col* coord_device, col* speed_device, long part_num, col* en_device)
{
	// Block index
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	float dt = 1.00e-12;
	col net_force, integral1, integral2;
	net_force.x = 0;
	net_force.y = 0;
	net_force.z = 0;
	integral1.x = 0;
	integral1.y = 0;
	integral1.z = 0;
	integral2.x = 0;
	integral2.y = 0;
	integral2.z = 0;
		
	float r = 0;
	for (int i=0; i < part_num; i++) {
		if (i != tx) {
			r = sqrt((coord_device[i].x-coord_device[tx].x) * (coord_device[i].x-coord_device[tx].x)+
			(coord_device[i].y-coord_device[tx].y) * (coord_device[i].y-coord_device[tx].y)+
			(coord_device[i].z-coord_device[tx].z) * (coord_device[i].z-coord_device[tx].z));
			if (r < 5) 
			{
				net_force.x += (coord_device[i].x-coord_device[tx].x)/r * tex1D(TableTex, r * 5000 + 0.5f);
				net_force.y += (coord_device[i].y-coord_device[tx].y)/r * tex1D(TableTex, r * 5000 + 0.5f);
				net_force.z += (coord_device[i].z-coord_device[tx].z)/r * tex1D(TableTex, r * 5000 + 0.5f);
			}	
			
		}
	}
	
	__syncthreads();
	coord_device[tx].x += (speed_device[tx].x * dt + net_force.x * dt * dt / 2); 
	coord_device[tx].y += (speed_device[tx].y * dt + net_force.y * dt * dt / 2); 
	coord_device[tx].z += (speed_device[tx].z * dt + net_force.z * dt * dt / 2); 
	speed_device[tx].x += (net_force.x * dt);  
	speed_device[tx].y += (net_force.y * dt);  
	speed_device[tx].z += (net_force.z * dt); 
	en_device[tx].x = (speed_device[tx].x * speed_device[tx].x + speed_device[tx].y * speed_device[tx].y + speed_device[tx].z * speed_device[tx].z); 


	if ( r < 5 ) en_device[tx].y = tex1D (PotTex, r * 5000 + 0.5f) ;
 	
   }



int main(int argc, char** argv)
{
	cudaSetDevice(7);
	long part_num = 0;
	char temp[1000];	
	
	TableTex.filterMode=cudaFilterModeLinear;
	//TableTex.normalized=1;
	PotTex.filterMode=cudaFilterModeLinear;
	//PotTex.normalized=1;
	col energy[1000]; //x - kinetic energy, y - potential, z - net energy
	for (int i = 0; i < 1000; i++) {
	energy[i].x=0;
	energy[i].y=0;
	energy[i].z=0;
	}
	
	//Reading interpolation table
	FILE *table, *potent;
	table = fopen( "table.txt", "r" );
	potent = fopen( "pot.txt", "r");
	long pr = 0;

	if( table != 0 )
	{
	
		while( fscanf(table, "%s ", temp) != EOF)
			pr++;
	}
	else
	{
		printf("particles.txt not found\n");
		return -1;
	}

	float *tab = (float*) malloc((size_t)sizeof(float) * pr);
	float *pot = (float*) malloc((size_t)sizeof(float) * pr);
	
	rewind(table);
	for (int i=0; i < pr; i++) {
	fscanf(table, "%e ", &tab[i]);
	fscanf(potent, "%e ", &pot[i]);
	}
	fclose(table);
	fclose(potent);
//1
	//Readying texture
	cudaArray * interpolation_table = NULL;                                   
	cudaArray * pot_table = NULL;

   	cudaChannelFormatDesc cfDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    	cudaMallocArray(&interpolation_table, &cfDesc, pr, 0);
  	cudaMallocArray(&pot_table, &cfDesc, pr, 0);
  	cudaMemcpyToArray ( interpolation_table, 0, 0, tab, pr * sizeof(float), cudaMemcpyHostToDevice );
  	cudaMemcpyToArray ( pot_table, 0, 0, pot, pr * sizeof(float), cudaMemcpyHostToDevice);
   	cudaBindTextureToArray(TableTex, interpolation_table);		
   	cudaBindTextureToArray(PotTex, pot_table);
   	
//end1
	//Reading particles' information
	FILE *part;
	part = fopen( "particles.txt", "r" );
	

	if( part != 0 )
	{
	
		while( fgets(temp, 1000, part) != NULL)
			part_num++;
	}
	else
	{
		printf("particles.txt not found\n");
		return -1;
	}
		
	col *coord = (col*) malloc((size_t)sizeof(col) * part_num);
	col *speed = (col*) malloc((size_t)sizeof(col) * part_num);
	col *en = (col*) malloc((size_t)sizeof(col) * part_num);
	rewind(part);
	for (int i=0; i < part_num * 3; i+=3) {
	fscanf(part, "%f ", &coord[i/3].x);
	fscanf(part, "%f ", &coord[i/3].y);
	fscanf(part, "%f\n", &coord[i/3].z);
	}
	fclose(part);
	
	col *coord_device, *speed_device, *en_device;
	cudaMalloc((void**)&coord_device, (size_t) sizeof(col) * part_num);
	cudaMalloc((void**)&speed_device, (size_t) sizeof(col) * part_num);	
	cudaMalloc((void**)&en_device, (size_t) sizeof(col) * part_num);
	cudaMemset(en_device, 0.0f, (size_t) sizeof(col) * part_num);		
	cudaMemset(coord_device, 0.0f, (size_t) sizeof(col) * part_num);
	cudaMemset(speed_device, 0.0f, (size_t) sizeof(col) * part_num);
	
	dim3 block (BLOCK_SIZE);
	dim3 grid (GRID_SIZE);
	for (int i = 0; i < part_num; i++) {
	speed[i].x=0;
	speed[i].y=0;
	speed[i].z=0;
	en[i].x=0;
	en[i].y=0;
	en[i].z=0;
	}


	for (int i = 0; i < 1000; i++ ) {
		cudaMemcpy(coord_device, coord, (size_t) sizeof(col) * part_num, cudaMemcpyHostToDevice);	
		cudaMemcpy(speed_device, speed, (size_t) sizeof(col) * part_num, cudaMemcpyHostToDevice);

		kernel<<<grid, block>>> (coord_device, speed_device, part_num, en_device);
		cudaThreadSynchronize();

		cudaMemcpy(coord, coord_device, (size_t) sizeof(col) * part_num, cudaMemcpyDeviceToHost);
		cudaMemcpy(speed, speed_device, (size_t) sizeof(col) * part_num, cudaMemcpyDeviceToHost);
		//printf("%f %f %f %f\n", coord[0].x, coord[0].y, coord[1].x, coord[1].y);
		//printf ("%f %f\n", coord[0].x, coord[0].y);
		cudaMemcpy(en, en_device, (size_t) sizeof(col) * part_num, cudaMemcpyDeviceToHost);
		for (int j=0; j < part_num; j++) { energy[i].x+=en[j].x; energy[i].y+=en[j].y;}
		energy[i].y*=0.5;
		energy[i].z=energy[i].x+energy[i].y;
		
	}

	cudaFree(coord_device);
	cudaFree(speed_device);
	cudaFree(en_device);
	free(coord);
	free(speed);
	free(en);
	cudaUnbindTexture(TableTex);
	cudaUnbindTexture(PotTex);
	cudaFree(interpolation_table);
	cudaFree(pot_table);
	//Writing results
	FILE *out;
	out = fopen( "out.txt", "w" ); 
	for (int  i = 0 ;i < 1000 ; i++)
	{
		fprintf(out, "%E\t%E\t%E\n", energy[i].x, energy[i].y, energy[i].z );
	}
	fclose( out );

  	return 0;
}
