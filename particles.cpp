//Code written by Richard O. Lee and Christian Bienia
//Modified by Christian Fensch

//////
#include "bTimer.hpp"
#include <unistd.h>
bTimer b;
//////


#include <cstdlib>
#include <cstring>

#include <iostream>
#include <fstream>
#include <math.h>
#include <assert.h>

#include "particles.hpp"
#include "cellpool.hpp"

#ifdef _OPENMP
   #include <omp.h>
#else
   #define omp_get_thread_num() 0
#endif

#ifdef ENABLE_VISUALIZATION
#include "view.hpp"
#endif

void CleanUpSim(void);                // freeing up allocated memory
void RebuildGrid(void);               // rebuilding the simulation's grid
void AdvanceFrame(void);              // calls functions in order to advance a frame of the simulation
void ComputeForces(void);             // computes the forces invoked in a frame
void ProcessCollisions(void);         // processes collision concerning the domainMax
void ProcessCollisions2(void);        // processes collision concerning the domainMin
void AdvanceParticles(void);          // after having computed all the necessary values, move the particles
void InitSim(char const *fileName);   // initializes the simulation
void SaveFile(char const *fileName);  // writes the results in a file
int GetNeighborCells(int ci, int cj, int ck, int *neighCells);  // gets the indexes of the neighbours in an array


//Uncomment to add code to check that Courant–Friedrichs–Lewy condition is satisfied at runtime
//#define ENABLE_CFL_CHECK

//Define ENABLE_STATISTICS to collect additional information about the particles at runtime.
//#define ENABLE_STATISTICS

////////////////////////////////////////////////////////////////////////////////

// pool of cells
cellpool pool;

// setting the simulation variables concerning the physics
float restParticlesPerMeter, h, hSq;
float densityCoeff, pressureCoeff, viscosityCoeff;
static float timeStep = 0.001;

int nx, ny, nz;               // number of grid cells in each dimension
Vec3 delta;                   // cell dimensions
int numParticles = 0;         // number of particles
int numCells = 0;             // number of cells
Cell *cells = NULL;           // array of cells
Cell *cells2 = NULL;          // helper array of cells
int *cnumPars = 0;            // array of particles
int *cnumPars2 = 0;           // helper array of particles
Cell **last_cells = NULL;     // helper array with pointers to last cell structure of "cells" array lists
omp_lock_t *cell_mutexes = 0; // array of mutexes for future use in race conditions

// visualization initial values
#ifdef ENABLE_VISUALIZATION
Vec3 vMax(0.0,0.0,0.0);
Vec3 vMin(0.0,0.0,0.0);
#endif

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Initializing the simulation's variables paying attention to the cache 
 * line's size depending on the system we are running on. The variables of the
 * simulation are initialized according to the given input file
 *
 * @param char const filename (input file with simulation parameters) 
 */
void InitSim(char const *fileName)
{

  // attempting to load the file
  std::cout << "Loading file \"" << fileName << "\"..." << std::endl;
  std::ifstream file(fileName, std::ios::binary);

    // file was not found
  if(!file) {
    std::cerr << "Error opening file. Aborting." << std::endl;
    exit(1);
  }

  // Always use single precision float variables b/c file format uses single precision
  // reading through the file's parameters paying attention to their memory organization
  float restParticlesPerMeter_le;
  int numParticles_le;

      // reading the first 2 values concerning the particles per meter and their total number
  file.read((char *)&restParticlesPerMeter_le, FILE_SIZE_FLOAT);
  file.read((char *)&numParticles_le, FILE_SIZE_INT);

      // if the data is in fact in big endian then swap them
  if(!isLittleEndian()) {
    restParticlesPerMeter = bswap_float(restParticlesPerMeter_le);
    numParticles          = bswap_int32(numParticles_le);
  } else {
    restParticlesPerMeter = restParticlesPerMeter_le;
    numParticles          = numParticles_le;
  }

  // initializing the the pool with cells
  cellpool_init(&pool, numParticles);

  // setting the height calculated by the kernel radius divided by restParticles per meter
  h = kernelRadiusMultiplier / restParticlesPerMeter; 
  hSq = h*h;  // height squared

  /* begin initializations doing it single threaded */
    
  // calculating the coefficients
  float coeff1 = 315.0 / (64.0*pi*powf(h,9.0));
  float coeff2 = 15.0 / (pi*powf(h,6.0));
  float coeff3 = 45.0 / (pi*powf(h,6.0));

  // particle's characteristics
  float particleMass = 0.5 * doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);
  densityCoeff = particleMass * coeff1;
  pressureCoeff = 3.0 * coeff2 * 0.50*stiffnessPressure * particleMass;
  viscosityCoeff = viscosity * coeff3 * particleMass;

  // setting the range according to the preset values
  Vec3 range = domainMax - domainMin;

  // setting the grid dimensions according to the range and height
  nx = (int)(range.x / h);
  ny = (int)(range.y / h);
  nz = (int)(range.z / h);

    // checking if all went well with the initialization
  assert(nx >= 1 && ny >= 1 && nz >= 1);

  // calculating the total number of cells in the simulation
  numCells = nx*ny*nz;
  std::cout << "Number of cells: " << numCells << std::endl;

  // setting the cells per dimension
  delta.x = range.x / nx;
  delta.y = range.y / ny;
  delta.z = range.z / nz;

    // checking if all went well with the initialization
  assert(delta.x >= h && delta.y >= h && delta.z >= h);

  /* making some cache adjustments to increace the hit ratio */
  // make sure Cell structure is multiple of estimated cache line size
  assert(sizeof(Cell) % CACHELINE_SIZE == 0);
  
  // make sure helper Cell structure is in sync with real Cell structure
  assert(offsetof(struct Cell_aux, padding) == offsetof(struct Cell, padding));

  // system aware allocations
#if defined(WIN32)
  cells = (struct Cell*)_aligned_malloc(sizeof(struct Cell) * numCells, CACHELINE_SIZE);
  cells2 = (struct Cell*)_aligned_malloc(sizeof(struct Cell) * numCells, CACHELINE_SIZE);
  cnumPars = (int*)_aligned_malloc(sizeof(int) * numCells, CACHELINE_SIZE);
  cnumPars2 = (int*)_aligned_malloc(sizeof(int) * numCells, CACHELINE_SIZE);
  last_cells = (struct Cell **)_aligned_malloc(sizeof(struct Cell *) * numCells, CACHELINE_SIZE);
  cell_mutexes = (omp_lock_t *)_aligned_malloc(sizeof(omp_lock_t) * numCells, CACHELINE_SIZE);
  assert((cells!=NULL) && (cells2!=NULL) && (cnumPars!=NULL) && (cnumPars2!=NULL) && (last_cells!=NULL) && (cell_mutexes!=NULL)); 
#else
  int rv0 = posix_memalign((void **)(&cells), CACHELINE_SIZE, sizeof(struct Cell) * numCells);
  int rv1 = posix_memalign((void **)(&cells2), CACHELINE_SIZE, sizeof(struct Cell) * numCells);
  int rv2 = posix_memalign((void **)(&cnumPars), CACHELINE_SIZE, sizeof(int) * numCells);
  int rv3 = posix_memalign((void **)(&cnumPars2), CACHELINE_SIZE, sizeof(int) * numCells);
  int rv4 = posix_memalign((void **)(&last_cells), CACHELINE_SIZE, sizeof(struct Cell *) * numCells);
  int rv5 = posix_memalign((void **)(&cell_mutexes), CACHELINE_SIZE, sizeof(omp_lock_t) * numCells);
  assert((rv0==0) && (rv1==0) && (rv2==0) && (rv3==0) && (rv4==0) && (rv5==0));
#endif

  // because cells and cells2 are not allocated via new
  // we construct them here
    // running the loop in parallel
    // each thread creates approximately equal number of cells
    // unless it the size is small enough to run single threaded
  if (numCells > 12000) {
    #pragma omp parallel for schedule(static)
    for(int i=0; i<numCells; ++i)
    {
      new (&cells[i]) Cell;
      new (&cells2[i]) Cell;

      // initializing the mutexes
      omp_init_lock(&cell_mutexes[i]);
    }
  } else {
    for(int i=0; i<numCells; ++i)
    {
      new (&cells[i]) Cell;
      new (&cells2[i]) Cell;

      // initializing the mutexes
      omp_init_lock(&cell_mutexes[i]);
    }
  }

  // setting the array elements to zero
  memset(cnumPars, 0, numCells*sizeof(int));

  // Always use single precision float variables b/c file format uses single precision
  float px, py, pz, hvx, hvy, hvz, vx, vy, vz;

  // parsing the file data  
  for(int i = 0; i < numParticles; ++i)
  {
      file.read((char *)&px, FILE_SIZE_FLOAT);
      file.read((char *)&py, FILE_SIZE_FLOAT);
      file.read((char *)&pz, FILE_SIZE_FLOAT);
      file.read((char *)&hvx, FILE_SIZE_FLOAT);
      file.read((char *)&hvy, FILE_SIZE_FLOAT);
      file.read((char *)&hvz, FILE_SIZE_FLOAT);
      file.read((char *)&vx, FILE_SIZE_FLOAT);
      file.read((char *)&vy, FILE_SIZE_FLOAT);
      file.read((char *)&vz, FILE_SIZE_FLOAT);

      // checking data order in memory
      if(!isLittleEndian()) {
        px  = bswap_float(px);
        py  = bswap_float(py);
        pz  = bswap_float(pz);
        hvx = bswap_float(hvx);
        hvy = bswap_float(hvy);
        hvz = bswap_float(hvz);
        vx  = bswap_float(vx);
        vy  = bswap_float(vy);
        vz  = bswap_float(vz);
      }

      // assigning grid positions
      int ci = (int)(((float)px - domainMin.x) / delta.x);
      int cj = (int)(((float)py - domainMin.y) / delta.y);
      int ck = (int)(((float)pz - domainMin.z) / delta.z);

      // checking if the grid positions are valid
      // if they are out of bound then reset them
      if(ci < 0) ci = 0; else if(ci >= nx) ci = nx-1;
      if(cj < 0) cj = 0; else if(cj >= ny) cj = ny-1;
      if(ck < 0) ck = 0; else if(ck >= nz) ck = nz-1;

      // calculate the index of the array depending 
      // on its grid position
      int index = (ck*ny + cj)*nx + ci;

      // get the position of this cell in the array
      Cell *cell = &cells[index];

      // add particle to cell
      // go to last cell structure in list
      int np = cnumPars[index];

      while(np > PARTICLES_PER_CELL) {
          cell = cell->next; 
          np -= PARTICLES_PER_CELL;
      } // while

      // add another cell structure if everything full
      if( (np % PARTICLES_PER_CELL == 0) && (cnumPars[index] != 0) ) {
        cell->next = cellpool_getcell(&pool);
        cell = cell->next;
        np -= PARTICLES_PER_CELL; // np = 0;
      }

      // setting this particle's coordinates in its cell
      cell->p[np].x = px;
      cell->p[np].y = py;
      cell->p[np].z = pz;
      cell->hv[np].x = hvx;
      cell->hv[np].y = hvy;
      cell->hv[np].z = hvz;
      cell->v[np].x = vx;
      cell->v[np].y = vy;
      cell->v[np].z = vz;

      // setting the coordinates for glut to see
    #ifdef ENABLE_VISUALIZATION
      vMin.x = std::min(vMin.x, cell->v[np].x);
      vMax.x = std::max(vMax.x, cell->v[np].x);
      vMin.y = std::min(vMin.y, cell->v[np].y);
      vMax.y = std::max(vMax.y, cell->v[np].y);
      vMin.z = std::min(vMin.z, cell->v[np].z);
      vMax.z = std::max(vMax.z, cell->v[np].z);
    #endif
        
        // increasing the particle counter for the computed index
      ++cnumPars[index];
  } // for

  std::cout << "Number of particles: " << numParticles << std::endl;
} // InitSim

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Save the output of the simulation to a file with the given name
 * 
 * @param char const *filename (output file name)
 */
void SaveFile(char const *fileName)
{
  std::cout << "Saving file \"" << fileName << "\"..." << std::endl;

  // attempting to open the file and checking if successful
  std::ofstream file(fileName, std::ios::binary);
  assert(file);

  // Always use single precision float variables b/c file format uses single precision
  // checking 
  if(!isLittleEndian()) {
    float restParticlesPerMeter_le;
    int   numParticles_le;

    restParticlesPerMeter_le = bswap_float((float)restParticlesPerMeter);
    numParticles_le      = bswap_int32(numParticles);
    file.write((char *)&restParticlesPerMeter_le, FILE_SIZE_FLOAT);
    file.write((char *)&numParticles_le,      FILE_SIZE_INT);
  } else {
    file.write((char *)&restParticlesPerMeter, FILE_SIZE_FLOAT);
    file.write((char *)&numParticles, FILE_SIZE_INT);
  }

  int count = 0;

  // iterate through the cells to write their data to a file
  for(int i = 0; i < numCells; ++i)
  {
    // get a pointer to the cell of index i
    Cell *cell = &cells[i];

    // get this cells number of particles
    int np = cnumPars[i];
   
    // iterate through all those particles 
    for(int j = 0; j < np; ++j)
    {
      //Always use single precision float variables b/c file format uses single precision
      float px, py, pz, hvx, hvy, hvz, vx,vy, vz;
     
      // check if we need to save as big endian
      if(!isLittleEndian()) {
        px  = bswap_float((float)(cell->p[j % PARTICLES_PER_CELL].x));
        py  = bswap_float((float)(cell->p[j % PARTICLES_PER_CELL].y));
        pz  = bswap_float((float)(cell->p[j % PARTICLES_PER_CELL].z));
        hvx = bswap_float((float)(cell->hv[j % PARTICLES_PER_CELL].x));
        hvy = bswap_float((float)(cell->hv[j % PARTICLES_PER_CELL].y));
        hvz = bswap_float((float)(cell->hv[j % PARTICLES_PER_CELL].z));
        vx  = bswap_float((float)(cell->v[j % PARTICLES_PER_CELL].x));
        vy  = bswap_float((float)(cell->v[j % PARTICLES_PER_CELL].y));
        vz  = bswap_float((float)(cell->v[j % PARTICLES_PER_CELL].z));
      } else {
        px  = (float)(cell->p[j % PARTICLES_PER_CELL].x);
        py  = (float)(cell->p[j % PARTICLES_PER_CELL].y);
        pz  = (float)(cell->p[j % PARTICLES_PER_CELL].z);
        hvx = (float)(cell->hv[j % PARTICLES_PER_CELL].x);
        hvy = (float)(cell->hv[j % PARTICLES_PER_CELL].y);
        hvz = (float)(cell->hv[j % PARTICLES_PER_CELL].z);
        vx  = (float)(cell->v[j % PARTICLES_PER_CELL].x);
        vy  = (float)(cell->v[j % PARTICLES_PER_CELL].y);
        vz  = (float)(cell->v[j % PARTICLES_PER_CELL].z);
      }

      // write the to the file
      file.write((char *)&px,  FILE_SIZE_FLOAT);
      file.write((char *)&py,  FILE_SIZE_FLOAT);
      file.write((char *)&pz,  FILE_SIZE_FLOAT);
      file.write((char *)&hvx, FILE_SIZE_FLOAT);
      file.write((char *)&hvy, FILE_SIZE_FLOAT);
      file.write((char *)&hvz, FILE_SIZE_FLOAT);
      file.write((char *)&vx,  FILE_SIZE_FLOAT);
      file.write((char *)&vy,  FILE_SIZE_FLOAT);
      file.write((char *)&vz,  FILE_SIZE_FLOAT);

      ++count;

      // move pointer to next cell in list if end of array is reached
      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell = cell->next;
      }
    }
  }

  // checking if we wrote as many particles as we read
  assert(count == numParticles);
}

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Iterating through the cells to clean up any memory allocations
 */
void CleanUpSim()
{
  // first return extended cells to cell pools
  #pragma omp parallel for ordered schedule(static, 100)
  for(int i=0; i<numCells; ++i)
  {
    Cell& cell = cells[i];
  	
    // iterate to the end of the cell list returning 
    // the cells to the pool
    while(cell.next)
  	{
  		Cell *temp = cell.next;
  		cell.next = temp->next;
  		cellpool_returncell(&pool, temp);
  	}

    // destroying locks
    omp_destroy_lock(&cell_mutexes[i]);
  }
  
  // now return cell pools
  cellpool_destroy(&pool);

  // free up all heap allocated space
  // by freeing all the arrays
#if defined(WIN32)
  _aligned_free(cells);
  _aligned_free(cells2);
  _aligned_free(cnumPars);
  _aligned_free(cnumPars2);
  _aligned_free(last_cells);
  _aligned_free(cell_mutexes);
#else
  free(cells);
  free(cells2);
  free(cnumPars);
  free(cnumPars2);
  free(last_cells);
  free(cell_mutexes);
#endif
}

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Rebuilds the grid by setting the positions of the particles
 * after having invoked forces and collisions on them
 */
void RebuildGrid()
{
  #pragma omp single
  {
    //swap src and dest arrays with particles
    std::swap(cells, cells2);
    
    //swap src and dest arrays with counts of particles
    std::swap(cnumPars, cnumPars2);

    //initialize destination data structures
    memset(cnumPars, 0, numCells*sizeof(int));
  }
  
  #pragma omp for schedule(static, 2)
  for(int i=0; i<numCells; i++)
  {
    cells[i].next = NULL;
    last_cells[i] = &cells[i];
  }

  #pragma omp single
  {
    // iterate through source cell lists
    for(int i = 0; i < numCells; ++i)
    {
      // get a pointer to the cell
      Cell *cell2 = &cells2[i];

      // get the number of particles of this cell
      int np2 = cnumPars2[i];

      // iterate through source particles
      for(int j = 0; j < np2; ++j)
      {
        // get destination for source particle
        int ci = (int)((cell2->p[j % PARTICLES_PER_CELL].x - domainMin.x) / delta.x);
        int cj = (int)((cell2->p[j % PARTICLES_PER_CELL].y - domainMin.y) / delta.y);
        int ck = (int)((cell2->p[j % PARTICLES_PER_CELL].z - domainMin.z) / delta.z);
      
          // confine to domain
          // Note, if ProcessCollisions() is working properly these tests are useless
        if(ci < 0) ci = 0; else if(ci >= nx) ci = nx-1;
        if(cj < 0) cj = 0; else if(cj >= ny) cj = ny-1;
        if(ck < 0) ck = 0; else if(ck >= nz) ck = nz-1;

  #ifdef ENABLE_CFL_CHECK
        // check that source cell is a neighbor of destination cell
        bool cfl_cond_satisfied=false;
        for(int di = -1; di <= 1; ++di)
          for(int dj = -1; dj <= 1; ++dj)
            for(int dk = -1; dk <= 1; ++dk)
            {
              int ii = ci + di;
              int jj = cj + dj;
              int kk = ck + dk;
              if(ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
              {
                int index = (kk*ny + jj)*nx + ii;
                if(index == i)
                {
                  cfl_cond_satisfied=true;
                  break;
                }
              }
            }
        if(!cfl_cond_satisfied)
        {
          std::cerr << "FATAL ERROR: Courant–Friedrichs–Lewy condition not satisfied." << std::endl;
          exit(1);
        }
  #endif //ENABLE_CFL_CHECK

        // get last pointer in correct destination cell list
        int index = (ck*ny + cj)*nx + ci;
        Cell *cell = last_cells[index];
        int np = cnumPars[index];

        // add another cell structure if everything full
        if( (np % PARTICLES_PER_CELL == 0) && (cnumPars[index] != 0) ) {
          cell->next = cellpool_getcell(&pool);
          cell = cell->next;
          last_cells[index] = cell;
        }
        ++cnumPars[index];

        // copy source to destination particle
        cell->p[np % PARTICLES_PER_CELL].x = cell2->p[j % PARTICLES_PER_CELL].x;
        cell->p[np % PARTICLES_PER_CELL].y = cell2->p[j % PARTICLES_PER_CELL].y;
        cell->p[np % PARTICLES_PER_CELL].z = cell2->p[j % PARTICLES_PER_CELL].z;
        cell->hv[np % PARTICLES_PER_CELL].x = cell2->hv[j % PARTICLES_PER_CELL].x;
        cell->hv[np % PARTICLES_PER_CELL].y = cell2->hv[j % PARTICLES_PER_CELL].y;
        cell->hv[np % PARTICLES_PER_CELL].z = cell2->hv[j % PARTICLES_PER_CELL].z;
        cell->v[np % PARTICLES_PER_CELL].x = cell2->v[j % PARTICLES_PER_CELL].x;
        cell->v[np % PARTICLES_PER_CELL].y = cell2->v[j % PARTICLES_PER_CELL].y;
        cell->v[np % PARTICLES_PER_CELL].z = cell2->v[j % PARTICLES_PER_CELL].z;

        // move pointer to next source cell in list if end of array is reached
        if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
          Cell *temp = cell2;
          cell2 = cell2->next;

          // return cells to pool that are not statically allocated head of lists
          if(temp != &cells2[i]) {
            cellpool_returncell(&pool, temp);
          }
        }

      } // j

      // return cells to pool that are not statically allocated head of lists
      if((cell2 != NULL) && (cell2 != &cells2[i])) {
        cellpool_returncell(&pool, cell2);
      }
    } // i
  }// omp single
}

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Stores the indexes of every neighbour of the given cell in
 * the array passed as parameter
 *
 * @param Grid coordinates of the cell (ci, cj, ck), neighCells
 * which is the array that will hold the neighbours' indexes
 */
int GetNeighborCells(int ci, int cj, int ck, int *neighCells)
{
  // counter for the neighbour cells
  int numNeighCells = 0;

  // have the nearest particles first -> help branch prediction
  int my_index = (ck*ny + cj)*nx + ci;  // getting my cell's index
  neighCells[numNeighCells] = my_index; // setting itself as a neighbour
  ++numNeighCells;                      // increasing the counter

  // checking around our cell (+- one position)
  for(int di = -1; di <= 1; ++di)
    for(int dj = -1; dj <= 1; ++dj)
      for(int dk = -1; dk <= 1; ++dk)
      {
        // offsets for the index (offsets around our index)
        int ii = ci + di;
        int jj = cj + dj;
        int kk = ck + dk;

        // checking if the offsets are within bounds
        if(ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
        {
          // getting the possible neighbour's index
          int index = (kk*ny + jj)*nx + ii;

          // checking if this is indeed a neighbour
          // and does have particles
          if((index < my_index) && (cnumPars[index] != 0))
          {
            // storing the index in the array
            neighCells[numNeighCells] = index;
            ++numNeighCells;  // increasin the neighbours' counter
          }
        }
      }// dk

  // return this counter !
  return numNeighCells;
}// GetNeighborCells

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Computing the forces that need to be applied to the particles
 * after each frame. Iterating through the entire grid of cells and all
 * their particles updating their densities as needed
 */
void ComputeForces()
{
    // initialize arrays, so that they are ready 
    // for updating 
    #pragma omp for schedule(static)
    for(int i = 0; i < numCells; ++i)
    {
      Cell *cell = &cells[i];
      int np = cnumPars[i];

      // for all the particles in the cell
      for(int j = 0; j < np; ++j)
      {
        // clear their density value
        cell->density[j % PARTICLES_PER_CELL] = 0.0;

        // set their accelaretion to the defaults
        cell->a[j % PARTICLES_PER_CELL] = externalAcceleration;

        // move pointer to next cell in list if end of array is reached
        if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
          cell = cell->next;
        }
      }
        
      // initialize the mutexes
      omp_init_lock(&cell_mutexes[i]);
    }

    // for all particles in the grid we will update their densities
    #pragma omp for ordered schedule(static) 
    for(int cindex=0; cindex<(nz*ny*nx); ++cindex)
    {
        // array of neigbours (counts them)
        int neighCells[27];

        // grid coordinates
        int ck = ( cindex-cindex % (nx*ny) ) / (ny*nx);
        int cj = ( (cindex-cindex%nx) / nx ) % ny;
        int ci = cindex % nx;

        // number of particles for this cell
        int np = cnumPars[cindex];

        // if np is zero there is no need to proceed,
        // we continue to the next iteration
        if(np == 0) {
          continue;
        }

        // get the neighbours and update the neighCells array
        int numNeighCells = GetNeighborCells(ci, cj, ck, neighCells);

        // lock the mutex for this cell
        omp_set_lock(&cell_mutexes[cindex]);

        // get a pointer to the cell we want to update
        Cell *cell = &cells[cindex];

        // for all the particles of this cell
        for(int ipar = 0; ipar < np; ++ipar)
        {
            // for all the neighbours
            for(int inc = 0; inc < numNeighCells; ++inc)
            {
                // get the neighbour's index
                int cindexNeigh = neighCells[inc];
                
                // if the neighbour is not our initial cell
                // then lock this critical section
                if (cindex != cindexNeigh) {
                  omp_set_lock(&cell_mutexes[cindexNeigh]);
                }   
                
                // get a pointer to the neighbour's cell
                Cell *neigh = &cells[cindexNeigh];

                // get the number of particles in the 
                // neighbour's cell
                int numNeighPars = cnumPars[cindexNeigh];

                // for all the paricles of the neighbour
                for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
                {
                    // Check address to make sure densities are computed only once per pair
                    if(&neigh->p[iparNeigh % PARTICLES_PER_CELL] < &cell->p[ipar % PARTICLES_PER_CELL])
                    {
                        // calculate the squared distance
                        float distSq = (cell->p[ipar % PARTICLES_PER_CELL] - neigh->p[iparNeigh % PARTICLES_PER_CELL]).GetLengthSq();
                        
                        // if it is valid according to the variable we
                        // set during the simulation's initialization
                        if(distSq < hSq)
                        {
                          // calculating the distance difference
                          float t = hSq - distSq;
                          float tc = t*t*t; 
             
                          // update both this cell's density and the neighbour's
                          cell->density[ipar % PARTICLES_PER_CELL] += (double)tc;
                          neigh->density[iparNeigh % PARTICLES_PER_CELL] += (double)tc;
                        }
                    }

                    // move pointer to next cell in list if end of array is reached
                    if(iparNeigh % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
                      neigh = neigh->next;
                    }
                }// iparNeigh

                // if we locked the mutex, then unlock now
                if (cindex != cindexNeigh) {
                  omp_unset_lock(&cell_mutexes[cindexNeigh]);
                }
             }

            // move pointer to next cell in list if end of array is reached
            if(ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {    
              cell = cell->next;
            }

        }// ipar

        // release the lock on this cell
        omp_unset_lock(&cell_mutexes[cindex]);

    }// cindex

    // calculate the tc 
    const float tc = hSq*hSq*hSq;
    #pragma omp for schedule(static)
    for(int i = 0; i < numCells; ++i)
    {
        // get a pointer to the current cell
        Cell *cell = &cells[i];

        // get the number of particles in the current cell
        int np = cnumPars[i];

        // for all the particles in the cell
        for(int j = 0; j < np; ++j)
        {
          // update the densities
          cell->density[j % PARTICLES_PER_CELL] += tc;
          cell->density[j % PARTICLES_PER_CELL] *= densityCoeff;

          // move pointer to next cell in list if end of array is reached
          if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
            cell = cell->next;
          }
        }
    }// i

    #pragma omp for ordered schedule(static) 
    for(int cindex=0; cindex<(nz*ny*nx); ++cindex)
    {
        // array of neigbours (counts them)
        int neighCells[27];

        // grid coordinates
        int ck = ( cindex-cindex % (nx*ny) ) / (ny*nx);
        int cj = ( (cindex-cindex%nx) / nx ) % ny;
        int ci = cindex % nx;

        // number of particles for this cell
        int np = cnumPars[cindex];

        // if np is zero there is no need to proceed,
        // we continue to the next iteration
        if(np == 0) {
          continue;
        }

        // get the neighbours and update the neighCells array
        int numNeighCells = GetNeighborCells(ci, cj, ck, neighCells);

        // lock the mutex for this cell
        omp_set_lock(&cell_mutexes[cindex]);
        
        // get a pointer to our cell
        Cell *cell = &cells[cindex];

        for(int ipar = 0; ipar < np; ++ipar)
        {
          for(int inc = 0; inc < numNeighCells; ++inc)
          {
                // get the neighbour's index
                int cindexNeigh = neighCells[inc];

                // if the neighbour is not our initial cell
                // then lock this critical section
                if (cindex != cindexNeigh) {
                  omp_set_lock(&cell_mutexes[cindexNeigh]);
                }   
                
                // get a pointer to the neighbour
                Cell *neigh = &cells[cindexNeigh];

                // get the number of the neighbour's particles
                int numNeighPars = cnumPars[cindexNeigh];
                
                for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
                {
                    // Check address to make sure forces are computed only once per pair
                    if(&neigh->p[iparNeigh % PARTICLES_PER_CELL] < &cell->p[ipar % PARTICLES_PER_CELL])
                    {
                        // calculate the distance between our particle and the neighbour's
                        Vec3 disp = cell->p[ipar % PARTICLES_PER_CELL] - neigh->p[iparNeigh % PARTICLES_PER_CELL];
                        
                        // get the distance squared
                        float distSq = disp.GetLengthSq();

                        // if it is valid according to the variable we
                        // set during the simulation's initialization         
                        if(distSq < hSq)
                        {
                            // get the distance (square root)
                            float dist = sqrtf(std::max(distSq, (float)1e-12));

                            // check the distance relative to the height
                            float hmr = h - dist;

                            // calculate the accelaration values we need to add to the current ones
                            Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) * ((float)cell->density[ipar % PARTICLES_PER_CELL]+(float)neigh->density[iparNeigh % PARTICLES_PER_CELL] - (float)doubleRestDensity);
                            
                            // update the accelaration values
                            acc += (neigh->v[iparNeigh % PARTICLES_PER_CELL] - cell->v[ipar % PARTICLES_PER_CELL]) * viscosityCoeff * hmr;
                            acc /= (float)cell->density[ipar % PARTICLES_PER_CELL] * (float)neigh->density[iparNeigh % PARTICLES_PER_CELL];

                            acc.x = (float)acc.x; acc.y = (float)acc.y; acc.z = (float)acc.z;

                            // set the cell's values to the ones we just calculated
                            cell->a[ipar % PARTICLES_PER_CELL] += acc;
                            neigh->a[iparNeigh % PARTICLES_PER_CELL] -= acc;
                        }
                    }

                    // move pointer to next cell in list if end of array is reached
                    if(iparNeigh % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
                      neigh = neigh->next;
                    }

                }// iparNeigh

                // if we locked the mutex, then unlock now
                if (cindex != cindexNeigh) {
                  omp_unset_lock(&cell_mutexes[cindexNeigh]);
                }
            }// inc

            // move pointer to next cell in list if end of array is reached
            if(ipar % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
              cell = cell->next;
            }
        }// ipar

        // release the lock on this cell
        omp_unset_lock(&cell_mutexes[cindex]);
    }// cindex
}// ComputeForces

////////////////////////////////////////////////////////////////////////////////
#if 0
/**
 * @brief Alternate process collision function with the below features:
 * ProcessCollisions() with container walls
 * Under the assumptions that
 * a) a particle will not penetrate a wall
 * b) a particle will not migrate further than once cell
 * c) the parSize is smaller than a cell
 * then only the particles at the perimiters may be influenced by the walls 
 */
void ProcessCollisions()
{
  #pragma omp for schedule(static)
	for(int i = 0; i < numCells; ++i)
	{
        Cell *cell = &cells[i];
        int np = cnumPars[i];

        for(int j = 0; j < np; ++j)
        {
          Vec3 pos = cell->p[j % PARTICLES_PER_CELL] + cell->hv[j % PARTICLES_PER_CELL] * timeStep;

          float diff = parSize - (pos.x - domainMin.x);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].x += stiffnessCollisions*diff - damping*cell->v[j % PARTICLES_PER_CELL].x;

          diff = parSize - (domainMax.x - pos.x);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].x -= stiffnessCollisions*diff + damping*cell->v[j % PARTICLES_PER_CELL].x;

          diff = parSize - (pos.y - domainMin.y);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].y += stiffnessCollisions*diff - damping*cell->v[j % PARTICLES_PER_CELL].y;

          diff = parSize - (domainMax.y - pos.y);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].y -= stiffnessCollisions*diff + damping*cell->v[j % PARTICLES_PER_CELL].y;

          diff = parSize - (pos.z - domainMin.z);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].z += stiffnessCollisions*diff - damping*cell->v[j % PARTICLES_PER_CELL].z;

          diff = parSize - (domainMax.z - pos.z);
          if(diff > epsilon)
            cell->a[j % PARTICLES_PER_CELL].z -= stiffnessCollisions*diff + damping*cell->v[j % PARTICLES_PER_CELL].z;

          //move pointer to next cell in list if end of array is reached
          if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1)
            cell = cell->next;
		}
	}
}
#else
// Notes on USE_ImpeneratableWall
// When particle is detected beyond cell wall it is repositioned at cell wall
// velocity is not changed, thus conserving momentum.
// What this means though it the prior AdvanceParticles had positioned the
// particle beyond the cell wall and thus the visualization will show these
// as artifacts. The proper place for USE_ImpeneratableWall is after AdvanceParticles.
// This would entail a 2nd pass on the perimiters after AdvanceParticles (as opposed
// to inside AdvanceParticles). Your fluid dynamisist should properly devise the
// equasions. 

/**
 * @brief Processing the particle collisions for each one of the dimensions
 * for domainMax
 */
void ProcessCollisions()
{
  // variables for the boundaries
  int x,y,z;

  // splitting the parallel region to independent sections
  #pragma omp sections
  { 
      #pragma omp section
      { 
          x=0; // along the domainMin.x wall

          // y axis min
          for(y=0; y<ny; ++y)
          {
              // elevating the z axis
              for(z=0; z<nz; ++z)
              {
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
                  for(int j = 0; j < np; ++j)
                  {
                      // getting particle position in the cell
                      int ji = j % PARTICLES_PER_CELL;

                      // calculating the new position and the distance
                      float pos_x = cell->p[ji].x + cell->hv[ji].x * timeStep;
                      float diff = parSize - (pos_x - domainMin.x);

                      // checking the tollerance
                      if(diff > epsilon) {
                        cell->a[ji].x += stiffnessCollisions*diff - damping*cell->v[ji].x;
                      }
                      
                      // move pointer to next cell in list if end of array is reached
                      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
                        cell = cell->next;
                      }
                  }// j
              }// z
          }// y
      }// section


      #pragma omp section
      { 
          x=nx-1; // along the domainMax.x wall

          for(y=0; y<ny; ++y)
          {
              for(z=0; z<nz; ++z)
              {
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];
                  
                  // iterating for all the particles
                  for(int j = 0; j < np; ++j)
                  {
                      // getting particle position in the cell
                      int ji = j % PARTICLES_PER_CELL;

                      // calculating the new position and the distance
                      float pos_x = cell->p[ji].x + cell->hv[ji].x * timeStep;
                      float diff = parSize - (domainMax.x - pos_x);

                      // checking the tollerance
                      if(diff > epsilon) {
                        cell->a[ji].x -= stiffnessCollisions*diff + damping*cell->v[ji].x;
                      }

                      // move pointer to next cell in list if end of array is reached
                      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
                        cell = cell->next;
                      }
                  }// j
              }// z
          }// y
      }// section
 

      #pragma omp section
      {
          y=0; // along the domainMin.y wall

          for(x=0; x<nx; ++x)
          {
              for(z=0; z<nz; ++z)
              {
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
                  for(int j = 0; j < np; ++j)
                  {
                      // getting particle position in the cell                    
                      int ji = j % PARTICLES_PER_CELL;

                      // calculating the new position and the distance
                      float pos_y = cell->p[ji].y + cell->hv[ji].y * timeStep;
                      float diff = parSize - (pos_y - domainMin.y);

                      // checking the tollerance
                      if(diff > epsilon) {
                        cell->a[ji].y += stiffnessCollisions*diff - damping*cell->v[ji].y;
                      }
                      
                      // move pointer to next cell in list if end of array is reached
                      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
                        cell = cell->next;
                      }
                  }// j
              }// z
          }// x
      }// section


      #pragma omp section
      {
          y=ny-1; // along the domainMax.y wall

          for(x=0; x<nx; ++x)
          {
              for(z=0; z<nz; ++z)
              {
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
                  for(int j = 0; j < np; ++j)
                  {
                      // getting particle position in the cell  
                      int ji = j % PARTICLES_PER_CELL;
                      
                      // calculating the new position and the distance
                      float pos_y = cell->p[ji].y + cell->hv[ji].y * timeStep;
                      float diff = parSize - (domainMax.y - pos_y);

                      // checking the tollerance
                      if(diff > epsilon) {
                        cell->a[ji].y -= stiffnessCollisions*diff + damping*cell->v[ji].y;
                      }

                      // move pointer to next cell in list if end of array is reached
                      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
                        cell = cell->next;
                      }
                  }// j
              }// z
          }// x
      }// section


      #pragma omp section
      {
          z=0; // along the domainMin.z wall
          for(x=0; x<nx; ++x)
          {
                for(y=0; y<ny; ++y)
                {
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
                  for(int j = 0; j < np; ++j)
                  {
                      // getting particle position in the cell  
                      int ji = j % PARTICLES_PER_CELL;

                      // calculating the new position and the distance
                      float pos_z = cell->p[ji].z + cell->hv[ji].z * timeStep;
                      float diff = parSize - (pos_z - domainMin.z);

                      // checking the tollerance
                      if(diff > epsilon) {
                        cell->a[ji].z += stiffnessCollisions*diff - damping*cell->v[ji].z;
                      }
                      
                      // move pointer to next cell in list if end of array is reached
                      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
                        cell = cell->next;
                      }
                  }// j
              }// y
          }// x
      }// section


      #pragma omp section
      {
          z=nz-1; // along the domainMax.z wall

          for(x=0; x<nx; ++x)
          {
              for(y=0; y<ny; ++y)
              {
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
                  for(int j = 0; j < np; ++j)
                  {
                      // getting particle position in the cell 
                      int ji = j % PARTICLES_PER_CELL;
                      
                      // calculating the new position and the distance
                      float pos_z = cell->p[ji].z + cell->hv[ji].z * timeStep;
                      float diff = parSize - (domainMax.z - pos_z);

                      // checking the tollerance
                      if(diff > epsilon) {
                        cell->a[ji].z -= stiffnessCollisions*diff + damping*cell->v[ji].z;
                      }

                      // move pointer to next cell in list if end of array is reached
                      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
                        cell = cell->next;
                      }
                  }// j
              }// y
          }// x
      }// section
  } // omp section
} // ProcessCollisions

#define USE_ImpeneratableWall
#if defined(USE_ImpeneratableWall)

/**
 * @brief Processing the particle collisions for each one of the dimensions
 * for domainMin
 */
void ProcessCollisions2()
{
	int x,y,z;

  #pragma omp sections
  {
      #pragma omp section
      {
        	x=0;	// along the domainMin.x wall
        	for(y=0; y<ny; ++y)
        	{
          		for(z=0; z<nz; ++z)
          		{
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
            			for(int j = 0; j < np; ++j)
            			{
                      // getting particle position in the cell 
              				int ji = j % PARTICLES_PER_CELL;

                      // calculating the distance between the cell and the domainMin
              				float diff = cell->p[ji].x - domainMin.x;

                      // checking if we hit a wall so that we reflect
                      // the particle by setting the oposite speeds
              				if(diff < Zero) {
              					cell->p[ji].x = domainMin.x - diff;
              					cell->v[ji].x = -cell->v[ji].x;
              					cell->hv[ji].x = -cell->hv[ji].x;
              				}

              				// move pointer to next cell in list if end of array is reached
              				if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
              					cell = cell->next;
                      }
            			}// j
          		}// z
        	}// y
      }// section

      #pragma omp section
      {
        	x=nx-1;	// along the domainMax.x wall
        	for(y=0; y<ny; ++y)
        	{
          		for(z=0; z<nz; ++z)
          		{
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
            			for(int j = 0; j < np; ++j)
            			{
                      // getting particle position in the cell
              				int ji = j % PARTICLES_PER_CELL;

                      // calculating the distance between the cell and the domainMin
              				float diff = domainMax.x - cell->p[ji].x;

                      // checking if we hit a wall so that we reflect
                      // the particle by setting the oposite speeds
              				if(diff < Zero) {
              					cell->p[ji].x = domainMax.x + diff;
              					cell->v[ji].x = -cell->v[ji].x;
              					cell->hv[ji].x = -cell->hv[ji].x;
              				}

              				// move pointer to next cell in list if end of array is reached
              				if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1)
              					cell = cell->next;
            			}// j
          		}// z
        	}// y
      }// section
    	
      #pragma omp section
      {
          y=0;	// along the domainMin.y wall
        	for(x=0; x<nx; ++x)
          {
            	for(z=0; z<nz; ++z)
            	{
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
            			for(int j = 0; j < np; ++j)
            			{
                      // getting particle position in the cell
              				int ji = j % PARTICLES_PER_CELL;

                      // calculating the distance between the cell and the domainMin
              				float diff = cell->p[ji].y - domainMin.y;

                      // checking if we hit a wall so that we reflect
                      // the particle by setting the oposite speeds
              				if(diff < Zero) {
              					cell->p[ji].y = domainMin.y - diff;
              					cell->v[ji].y = -cell->v[ji].y;
              					cell->hv[ji].y = -cell->hv[ji].y;
              				}

              				// move pointer to next cell in list if end of array is reached
              				if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1)
              					cell = cell->next;
            			}// j
            	}// z
        	 }// x
      }// section

      #pragma omp section
      {
          y=ny-1;	// along the domainMax.y wall
          for(x=0; x<nx; ++x)
          {
            	for(z=0; z<nz; ++z)
            	{
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
              		for(int j = 0; j < np; ++j)
              		{
                      // getting particle position in the cell
                			int ji = j % PARTICLES_PER_CELL;

                      // calculating the distance between the cell and the domainMin
                			float diff = domainMax.y - cell->p[ji].y;

                      // checking if we hit a wall so that we reflect
                      // the particle by setting the oposite speeds
                			if(diff < Zero) {
                				cell->p[ji].y = domainMax.y + diff;
                				cell->v[ji].y = -cell->v[ji].y;
                				cell->hv[ji].y = -cell->hv[ji].y;                          
                      }

                			// move pointer to next cell in list if end of array is reached
                			if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1)
                				cell = cell->next;
              		}// j
            	}// z
          }// x
      }// section

      #pragma omp section
      {
        	z=0;	// along the domainMin.z wall
        	for(x=0; x<nx; ++x)
        	{
          		for(y=0; y<ny; ++y)
          		{
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
            			for(int j = 0; j < np; ++j)
            			{
                      // getting particle position in the cell
              				int ji = j % PARTICLES_PER_CELL;

                      // calculating the distance between the cell and the domainMin
              				float diff = cell->p[ji].z - domainMin.z;

                      // checking if we hit a wall so that we reflect
                      // the particle by setting the oposite speeds
              				if(diff < Zero) {
              					cell->p[ji].z = domainMin.z - diff;
              					cell->v[ji].z = -cell->v[ji].z;
              					cell->hv[ji].z = -cell->hv[ji].z;
              				}

              				// move pointer to next cell in list if end of array is reached
              				if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1)
              					cell = cell->next;
            			}// j
          		}// y
        	}// x
      }// section

      #pragma omp section
      {
        	z=nz-1;	// along the domainMax.z wall
        	for(x=0; x<nx; ++x)
        	{
          		for(y=0; y<ny; ++y)
          		{
                  // calculating the corresponding index
                  int ci = (z*ny + y)*nx + x;

                  // getting a pointer to this cell
                  Cell *cell = &cells[ci];

                  // getting the number of particles
                  int np = cnumPars[ci];

                  // iterating for all the particles
            			for(int j = 0; j < np; ++j)
            			{
                      // getting particle position in the cell
              				int ji = j % PARTICLES_PER_CELL;

                      // calculating the distance between the cell and the domainMin
              				float diff = domainMax.z - cell->p[ji].z;

                      // checking if we hit a wall so that we reflect
                      // the particle by setting the oposite speeds
              				if(diff < Zero) {
              					cell->p[ji].z = domainMax.z + diff;
              					cell->v[ji].z = -cell->v[ji].z;
              					cell->hv[ji].z = -cell->hv[ji].z;
              				}

              				// move pointer to next cell in list if end of array is reached
              				if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1)
              					cell = cell->next;
            			}// j
          		}// y
        	}// x
      }// section
  }// sections
}// ProcessCollisions2
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Advancing the particles' positions and velocities depending on
 * their new status
 */
void AdvanceParticles()
{
  Cell *cell; // pointer to a cell
  int np;     // number of particles in a cell
  int j;      // inner for counter

  // iterating through the cells
  // moving them accordingly
  #pragma omp for schedule(static, 2) private(cell, np, j)
  for(int i = 0; i < numCells; ++i)
  {
    cell = &cells[i];
    np = cnumPars[i];

    for(j = 0; j < np; ++j)
    {
      // computing the new velocity
      Vec3 v_half = cell->hv[j % PARTICLES_PER_CELL] + cell->a[j % PARTICLES_PER_CELL]*timeStep;

#if defined(USE_ImpeneratableWall)
#endif

      // updating the cell's values (position, velocity, etc)
      cell->p[j % PARTICLES_PER_CELL] += v_half * timeStep;
      cell->v[j % PARTICLES_PER_CELL] = cell->hv[j % PARTICLES_PER_CELL] + v_half;
      cell->v[j % PARTICLES_PER_CELL] *= 0.5;
      cell->hv[j % PARTICLES_PER_CELL] = v_half;

      // move pointer to next cell in list if end of array is reached
      if(j % PARTICLES_PER_CELL == PARTICLES_PER_CELL-1) {
        cell = cell->next;
      }

    }// j
  }// i
}// AdvanceParticles

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Calls the below functions in order to build the grid, compute the 
 * forces in the fluid, process the collisions and move the particles around.
 * This function is the one that holds the simulation together by calling the 
 * appropriate functions
 */
void AdvanceFrame()
{
  #pragma omp parallel
  {
    RebuildGrid();
    ComputeForces();
    ProcessCollisions();
    AdvanceParticles();

  #if defined(USE_ImpeneratableWall)
    // N.B. The integration of the position can place the particle
    // outside the domain. We now make a pass on the perimeter cells
    // to account for particle migration beyond domain.
    ProcessCollisions2();
  #endif

  #ifdef ENABLE_STATISTICS
    float mean, stddev;
    int i;

    #pragma omp master
    {
      mean = (float)numParticles/(float)numCells;
      stddev = 0.0;
    }

    #pragma omp for schedule(static)
    for(i=0; i<numCells; ++i) {
      #pragma omp atomic
      stddev += (mean-cnumPars[i])*(mean-cnumPars[i]);
    }

    #pragma omp single
    {
      stddev = sqrtf(stddev);
      std::cout << "Cell statistics: mean=" << mean << " particles, stddev=" << stddev << " particles." << std::endl;
    }
  #endif
  }// omp parallel
}// AdvanceFrame

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief main, responsible for calling the initialization functions and 
 * the ones responsible for the simulation
 */
int main(int argc, char *argv[])
{
  // checking argument count and if wrong printing a usage message
  if(argc < 3 || argc >= 5) {
    std::cout << "Usage: " << argv[0] << " <framenum> <.fluid input file> [.fluid output file]" << std::endl;
    return -1;
  }

  // parsing to int the frame number given by the user
  int framenum = atoi(argv[1]);

  // wrong frame number, terminating ...
  if(framenum < 1) {
    std::cerr << "<framenum> must at least be 1" << std::endl;
    return -1;
  }

  // ENABLE_CFL_CHECK was enabled (Courant–Friedrichs–Lewy condition)
#ifdef ENABLE_CFL_CHECK
  std::cout << "WARNING: Check for Courant–Friedrichs–Lewy condition enabled. Do not use for performance measurements." << std::endl;
#endif

  // initializing the simulation with the given input file
  InitSim(argv[2]);

  // visualization enabled at compile time, setting the handlers with glut
#ifdef ENABLE_VISUALIZATION
  InitVisualizationMode(&argc, argv, &AdvanceFrame, &numCells, &cells, &cnumPars);
#endif

  // if no visualization was enabled
#ifndef ENABLE_VISUALIZATION

  // core of benchmark program (the Region-of-Interest)
  for(int i = 0; i < framenum; ++i)
    AdvanceFrame();

  // else render the new data
#else
  Visualize();
#endif

  // if requested by the user, then save the outputs to a file
  if(argc > 3)
    SaveFile(argv[3]);

  // cleaning up memory allocations
  CleanUpSim();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////

