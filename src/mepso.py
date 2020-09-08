import numpy as np
import math as mt
import random as rnd

def norm_two(v1,v2):
  r = np.float64(0.0)
  for i in range(len(v1)):
    r += pow(v1[i]-v2[i],2)
  return mt.sqrt(r)

def h(c1,c2):

  h1 = c1[0]*360
  h2 = c2[0]*360

  diff_abs = mt.fabs(h1-h2)
  h_dist = min(diff_abs, 360 - diff_abs) / 180

  if h1 == mt.nan or h2 == mt.nan:
    d = np.float64(0.0)
  else:
    d = h_dist

  return pow(d,2)


def hsl(c1,c2):
  h1 = c1[0]*360
  h2 = c2[0]*360

  s1 = c1[1]
  s2 = c2[1]

  l1 = c1[2]
  l2 = c2[2]

  diff_abs = mt.fabs(h1-h2)
  h_dist = min(diff_abs, 360 - diff_abs) / 180

  s1 *= mt.pi / 2
  s2 *= mt.pi / 2

  l_dist = mt.fabs(l1 - l2)

  c1 = mt.sin(s1)
  c2 = mt.sin(s2)

  a1 = mt.cos(s1)
  a2 = mt.cos(s2)

  if h1 == mt.nan or h2 == mt.nan:
    d = l_dist
  else:
    d = (a1 * a2 * l_dist) + (c1 * c2 * h_dist)

  return pow(d,2)



class MEPSO:
  ''' This PSO version using k-means fuzzy clustering of color to get a 
      segmentation of the image, where the color of the segmentation
      is the color of the centroid
  '''

  @staticmethod
  def _get_int_rnd(min,max):
    return rnd.randint(min,max)

  @staticmethod
  def _get_float_rnd(min,max):
    return rnd.uniform(min,max)

  def __init__( self,img,width,height,pop_size,cmin,cmax,type_color,
                thres_act=0.5,c1=1,c2=1,inertia=None,size_window=5,distance='rgb',epochs=1000 ):

    self.pop_size       = pop_size       #number of particles
    self.type_color     = type_color     #type color to process
    self.epochs         = epochs         #stop iter condition
    self.size_window    = size_window    #size spatial windows
    self.T              = thres_act      #activation threshold
    self.dims           = 3              #dimension of a image RGB, HSL
    self.inertia        = inertia        #inertia or weight value
    
    #distance metric
    if distance == 'hsl':
      self.distance = hsl
    elif distance == 'h':
      self.distance = h
    elif  distance == 'rgb':
      self.distance = norm_two
    else:
      print('Error dostance setting: must provide a valid metric')
      raise ValueError



    #image properties
    self.img          = img
    self.width        = width
    self.height       = height

    #aceleration constants 
    self.c1           = c1
    self.c2           = c2

    #min & max number of clusters
    self.cmin         = cmin
    self.cmax         = cmax

    if self.type_color == 'RGB':
      
      self.type = int
      self.get_random = MEPSO._get_int_rnd

      #particle velocities
      self.vmax     = 255
      self.vmin     = -255

      #particle val coordinate
      self.coormin  = 0
      self.coormax  = 255

    elif self.type_color == 'HSL':

      self.type = np.float64
      self.get_random = MEPSO._get_float_rnd

      #particle velocities
      self.vmax     = 1
      self.vmin     = -1

      #particle val coordinate
      self.coormin  = 0.0
      self.coormax  = 1.0

    else:
      print('Error color setting: must provide RBG or HSL opcion color')
      raise ValueError

    #particle activation coordinates
    self.amax       = 1
    self.amin       = 0

    #particle activation coordinates
    self.avmax      = 1
    self.avmin      = -1
  
  '''Initialization methods'''

  def _init_particles(self):
    '''Random centroids and velocities for init population of particles'''

    self.particles  = np.zeros((self.pop_size,self.cmax,self.dims),dtype=self.type)
    self.velocities = np.zeros((self.pop_size,self.cmax,self.dims),dtype=self.type)

    for i in range(self.pop_size):
      for j in range(self.cmax):
        for k in range(self.dims):
          self.particles[i][j][k] = self.get_random(self.coormin,self.coormax)
          self.velocities[i][j][k] = self.get_random(self.vmin,self.vmax)

  def _init_activations(self):
    '''Random centroids and velocities (activations) for init population of particles'''

    self.vel_activations  = np.zeros((self.pop_size,self.cmax),dtype=float)
    self.activations      = np.zeros((self.pop_size,self.cmax),dtype=float)
    
    for i in range(self.pop_size):
      for j in range(self.cmax):
        self.activations[i][j]     = rnd.uniform(self.amin,self.amax)
        self.vel_activations[i][j] = rnd.uniform(self.avmin,self.avmax)

  '''Fitness methods'''

  def spatial_windows(self,c1,c2,u_matrix):
    '''Calculate the neighborhood of a center data point'''

    h = 0

    lmin = mt.floor(self.size_window/2)
    lmax = lmin+1

    for i in range(c1-lmin,c1+lmax):
      if i >= 0 and i < self.height:
        for j in range(c2-lmin,c2+lmax):
          if j >= 0 and j < self.width:
            h += u_matrix[i][j]

    return h

  def degree_membership(self,c1,c2,particle,act,c,m=2):
    ''' Get Umatrix (degree of membership of point to centroid)'''

    dem  = 0
    point = self.img[c1][c2]
    diff = mt.fabs(self.distance(point,particle[c]))

    for i in range(self.cmax):
      if act[i] >= self.T:
        c = mt.fabs(self.distance(point,particle[i]))
        if c == 0:
          c = 0.000001
        dem += pow(diff/c,2/(m-1))

    if dem == 0:
      dem = 0.000001

    return 1 / dem

  def get_Umatrix(self,particle,act,c):

    Umatrix = np.zeros((self.height,self.width))

    for i in range(self.height):
      for j in range(self.width):
        Umatrix[i][j] = self.degree_membership(i,j,particle,act,c)

    return Umatrix

  def min_dist(self,id):

    min = mt.inf
    act = self.activations[id]
    particle = self.particles[id]

    for i in range(self.cmax):
      if act[i] >= self.T:
        p1 = particle[i]
        for j in range(self.cmax-i-1):
          print(i,j+i+1)
          if act[j+i+1] >= self.T:
            p2 = particle[j+i+1]
            dist = pow(self.distance(p1,p2),2)
            if dist < min:
              min = dist
    return min 

  def xie_beni(self,id):
    '''Include in this calculations:
       - First calculate all U-matrix and H-matrix for each valid cluster
       - Calculate U'-matrix to calculate degree of points to clusters,
         and the number of points for each valid cluster
       - Check valid cluster
       - Recalculate U'-matrix if some change is made, and calculate numerator
       - Calculate min dist for denominator
    '''

    valid_acts = []
    
    for i in range(self.cmax):
      if self.activations[id][i] >= self.T:
        valid_acts.append(i)

    size_valid = len(valid_acts)

    Utensor = np.zeros((size_valid,self.height,self.width))
    Htensor = np.zeros((size_valid,self.height,self.width))
    mglobal = np.zeros((self.height,self.width))

    index = 0

    for i in valid_acts:

      Utensor[index] = self.get_Umatrix(self.particles[id],self.activations[id],i)
      
      for x in range(self.height):
        for y in range(self.width):
          Htensor[index][x][y] = self.spatial_windows(x,y,Utensor[index])
          mglobal[x][y] += Utensor[index][x][y] * Htensor[index][x][y]

      index += 1

    '''Calculate U'-matrix to be the numerator for the method'''

    points = np.zeros(len(valid_acts),dtype=int)

    for x in range(self.height):
      for y in range(self.width):

        max_u = 0
        id_max = 0
        index = 0

        for i in valid_acts:
  
          centroid = self.particles[id][i]
        
          new_u = Utensor[index][x][y]*Htensor[index][x][y] / mglobal[x][y]
          if new_u > max_u:
            max_u = new_u
            id_max = index

          index += 1

        points[id_max] += 1

    '''Checking valid clusters have min points, else change act val or also reinitializate'''

    vals = len(valid_acts)
    print(vals)

    for i in range(vals):

      if points[i] == 0:

        if vals == self.cmin:
        
          #correction one cluster point
          pos = valid_acts[i]
          self.activations[id][pos] = 0.51
          for j in range(self.dims):
            self.particles[id][pos][j] = self.get_random(self.coormin,self.coormax)
        
          mglobal -= Utensor[i]*Htensor[i] #correct global sum of h's and u's
          #calculate news matrices U and H
          Utensor[i] = self.get_Umatrix(self.particles[id],self.activations[id],pos)
        
          for x in range(self.width):
            for y in range(self.height):
              Htensor[i][x][y] = self.spatial_windows(x,y,Utensor[i])
              mglobal[x][y] += Utensor[i][x][y] * Htensor[i][x][y]

        else:

          self.activations[id][valid_acts[i]] = 0.49
          vals -= 1

      elif points[i] == 1:

        #correction one cluster point
        pos = valid_acts[i]
        self.activations[id][pos] = 0.51
        for j in range(self.dims):
          self.particles[id][pos][j] = self.get_random(self.coormin,self.coormax)
       
        mglobal -= Utensor[i]*Htensor[i] #correct global sum of h's and u's
        #calculate news matrices U and H
        Utensor[i] = self.get_Umatrix(self.particles[id],self.activations[id],pos)
      
        for x in range(self.width):
          for y in range(self.height):
            Htensor[i][x][y] = self.spatial_windows(x,y,Utensor[i])
            mglobal[x][y] += Utensor[i][x][y] * Htensor[i][x][y]
    


    '''Calculate new U'matrix'''

    num = 0

    for x in range(self.width):
      for y in range(self.height):

        max_u = 0
        id_max = 0
        index = 0

        for i in valid_acts:
  
          centroid = self.particles[id][i]
        
          new_u = Utensor[index][x][y]*Htensor[index][x][y] / mglobal[x][y]

          point = self.img[x][y]
          dist = pow(self.distance(point,centroid),2)

          num += dist * pow(new_u,2)

          index += 1

    '''Get min distance between valid clusters a.k.a denominator'''

    n = self.width * self.height
    dem = n * self.min_dist(id)
    print(num,dem)

    return num / dem    

  def fitness(self):

    vec_fitness = np.zeros(self.pop_size)

    for i in range(self.pop_size):
      xie_beni = self.xie_beni(i)
      if xie_beni == 0:
        print('Xie-beni = 0')
        print(self.particles[i])
        print(self.img)
      vec_fitness[i] = 1 / (xie_beni + 0.0001)

    self.vec_fitness = vec_fitness

  '''Update velocities (gradients)'''

  def _update_velocities(self,file):

    if self.inertia != None:
      inertia = self.inertia
    else:
      inertia = rnd.uniform(0,1)

    pBest = self.pos_best_global

    for p in range(self.pop_size):

      r1 = rnd.uniform(0,self.c1)
      r2 = rnd.uniform(0,self.c2)

      pCur = self.particles[p]
      pLocalBest = self.pos_local_best[p]

      next_gradient = self.velocities[p] * inertia
      local_influence = r1 * (pLocalBest - pCur)
      global_influence = r2 * (pBest - pCur)

      self.velocities[p] = next_gradient + local_influence + global_influence 

      ##write on file
      file.write(f'Particle {p+1}\n')
      file.write(f'r1 = {r1:.6f}, r2 = {r2:.6f}\n')
      file.write(f'{self.velocities[p]}\n')
      ############################################

  def _update_vel_activations(self,file):

    if self.inertia != None:
      inertia = self.inertia
    else:
      inertia = rnd.uniform(0,1)

    pBest = self.pos_best_act_global

    for p in range(self.pop_size):

      r1 = rnd.uniform(0,self.c1)
      r2 = rnd.uniform(0,self.c2)

      pCur = self.activations[p]
      pLocalBest = self.pos_local_act_best[p]

      next_gradient = self.vel_activations[p] * inertia
      local_influence = r1 * (pLocalBest - pCur)
      global_influence = r2 * (pBest - pCur)

      self.vel_activations[p] = next_gradient + local_influence + global_influence  

      ##write on file
      file.write(f'Particle {p+1} Activations\n')
      file.write(f'r1 = {r1:.6f}, r2 = {r2:.6f}\n')
      file.write(f'{self.vel_activations[p]}\n')
      ############################################

  '''Checking functions'''

  def check_activations(self):

    for i in range(self.pop_size):
      for j in range(self.cmax):
        c = self.activations[i][j] 
        if c < self.amin:
          self.activations[i][j] = self.amin
        elif c > self.amax:
          self.activations[i][j] = self.amax

  def check_limit_particles(self):

    for i in range(self.pop_size):
      for j in range(self.cmax):
        for k in range(self.dims):
          if self.particles[i][j][k] < self.coormin:
            self.particles[i][j][k] = self.coormin
          elif self.particles[i][j][k] > self.coormax:
            self.particles[i][j][k] = self.coormax

  def check_valid_particles(self):

    for i in range(self.pop_size):
      invalid = []

      for j in range(self.cmax):
        if self.activations[i][j] < self.T:
          invalid.append(j)
      
      while self.cmax - len(invalid) < self.cmin:

        id = rnd.randint(0,len(invalid)-1)
        self.activations[i][invalid[id]] = rnd.uniform(0.5,self.amax)
        for k in range(self.dims): 
          self.particles[i][invalid[id]][k] = self.get_random(self.coormin,self.coormax)
        invalid.pop(id)

  '''Format write on output file'''

  def write_pairs(self,file,vec1,vec2,name1,name2):
    '''name1 - name2'''
    no = 'no'

    file.write(f'|{no:^3}|')
    for i in range(self.dims):
      dim = f'{name1}{i+1}'
      file.write(f'{dim:^7}|')

    file.write(f'{name2:^7}|\n')

    for i in range(self.pop_size):
      for j in range(self.cmax):
        file.write(f'|{i+1:^3}|')
        for k in range(self.dims):
          file.write(f'{vec1[i][j][k]:^7.2f}|')
        file.write(f'{vec2[i][j]:^7.2f}|')
        file.write('\n')

  def write_fitness(self,file,vec):
    '''name1 - name2'''
    no = 'no'
    name = 'fitness'

    file.write(f'|{no:^3}|')
    file.write(f'{name:^7}|\n')

    for i in range(self.pop_size):
      file.write(f'|{i+1:^3}|')
      file.write(f'{vec[i]:^7.4f}|')
      file.write('\n')


  '''Training'''

  def train(self):

    # file open #######################################################
    file = open('results.txt','a')
    file.write('Parameters\n')
    file.write(f'- Velocities between {self.vmin} and {self.vmax}\n')
    file.write(f'- Inertia (W) = {self.inertia}\n')
    file.write(f'- r1 and r2 values between {self.vmin} and {self.vmax}\n')
    file.write(f'- c1 = {self.c1} and c2 = {self.vmax}\n')
    file.write(f'- Epochs = {self.epochs}\n')
    file.write('\n')
    ###################################################################

    '''Init population process'''
    self._init_particles()
    self._init_activations()

    self.vec_fitness = np.zeros(self.pop_size)
    
    self.check_valid_particles()
    self.check_limit_particles()
    
    last_fitness_particles = np.zeros(self.pop_size)
    
    '''Init betas (growth ratio) for multi-elitist strategy'''
    self.growth_rates = np.zeros(self.pop_size,dtype=int)

    self.fitness() #get fitness population

    '''Init local best particles'''

    self.pos_local_best     = np.copy(self.particles) # local best particle position
    self.pos_local_act_best = np.copy(self.activations)
    self.local_best         = np.copy(self.vec_fitness)

    '''Get global best particle'''
    id_best_global = np.argmax(self.vec_fitness) 
    self.best_global = np.copy(self.vec_fitness[id_best_global])
    self.pos_best_global = np.copy(self.particles[id_best_global])
    self.pos_best_act_global = np.copy(self.activations[id_best_global])

    print(f'best global = {self.best_global:.5f}')

    #write particles on file
    #######################################################################
    file.write('Init Population\n')
    self.write_pairs(file,self.particles,self.activations,'dim','act')
    file.write('\n')
    file.write('Init Velocities\n')
    self.write_pairs(file,self.velocities,self.vel_activations,'dim','act')
    file.write('\n')
    file.write('Fitness of particles\n')
    self.write_fitness(file,self.vec_fitness)
    file.write('\n')
    file.write('Best global so far\n')
    file.write(f'particle {id_best_global} = {self.best_global}\n')
    file.write('\n')
    #######################################################################

    old_global = 0
    epoch = 0

    while mt.fabs(self.best_global - old_global) > 0.000001 and epoch < self.epochs:

      #write number of iteration ########################################
      file.write('\n')
      file.write(f'Iteration {epoch+1}\n')
      file.write(f'w = {self.inertia}\n')
      file.write('\n')
      ###################################################################

      last_fitness_particles = np.copy(self.vec_fitness)

      '''Mutation of particles'''
      self._update_velocities(file)
      self._update_vel_activations(file)
      self.particles    += self.velocities
      self.activations  += self.vel_activations

      '''Check valid activations and valid particle with minima cmin'''
      self.check_activations()
      self.check_valid_particles()
      self.check_limit_particles()

      self.fitness()

      '''Multi-etilist strategy to find local & best particles'''

      #candidate_area = []
      mx = -1

      for i in range(self.pop_size):

        fit = np.copy(self.vec_fitness[i])

        if last_fitness_particles[i] < fit:
          
          self.growth_rates[i] += 1 #update beta value
          self.local_best[i] = fit #update local best
          self.pos_local_best[i] = np.copy(self.particles[i])
          self.pos_local_act_best[i] = np.copy(self.activations[i]) #update best local activations

        if self.local_best[i] > self.best_global:

          #candidate_area.append(i) #using index to save candidate

          if self.growth_rates[i] > mx:

            mx = self.growth_rates[i]
            id_max = i

      #if len(candidate_area) > 0:
      if mx != -1:

        #print('pass')
        #best_candidate = 0
        #id = 0

        #for index in candidate_area:

          #if self.growth_rates[index] == mx and best_candidate < self.vec_fitness[index]:

            #best_candidate = self.vec_fitness[index]
            #id = index
        
        old_global = np.copy(self.best_global)

        self.best_global = self.local_best[id_max]
        self.pos_best_global = np.copy(self.pos_local_best[id_max])
        self.pos_best_act_global = np.copy(self.pos_local_act_best[id_max])

      epoch += 1

      #write on file best local ############################################################
      file.write('Best locales\n')
      self.write_pairs(file,self.pos_local_best,self.pos_local_act_best,'dim','act')
      file.write('\n')
      file.write('Fitness of particles\n')
      self.write_fitness(file,self.local_best)
      file.write('\n')
      file.write('Best global so far\n')
      file.write(f'particle {self.best_global} = {self.best_global}\n')
      #######################################################################################

      print(f'|{epoch}/{self.epochs}| best global = {self.best_global:.5f} , error = {mt.fabs(self.best_global - old_global):.5f}')
    
    file.close()

  '''Show result'''

  def get_image(self):

    valid_acts = []
    
    for i in range(self.cmax):
      if self.pos_best_act_global[i] >= self.T:
        valid_acts.append(i)

    size_valid = len(valid_acts)

    Utensor = np.zeros((size_valid,self.height,self.width))
    Htensor = np.zeros((size_valid,self.height,self.width))
    mglobal = np.zeros((self.height,self.width))

    index = 0

    for i in valid_acts:

      Utensor[index] = self.get_Umatrix(self.pos_best_global,self.pos_best_act_global,i)
      
      for x in range(self.width):
        for y in range(self.height):
          Htensor[index][x][y] = self.spatial_windows(x,y,Utensor[index])
          mglobal[x][y] += Utensor[index][x][y] * Htensor[index][x][y]

      index += 1

    '''Calculate U'-matrix to be the numerator for the method'''

    points = np.zeros((self.height,self.width,self.dims),dtype=self.type)

    for x in range(self.height):
      for y in range(self.width):

        max_u = 0
        id_max = 0
        index = 0

        for i in valid_acts:
        
          new_u = Utensor[index][x][y]*Htensor[index][x][y] / mglobal[x][y]
          if new_u > max_u:
            max_u = new_u
            id_max = index

          index += 1

        points[x][y] = self.pos_best_global[id_max]

    return points
       