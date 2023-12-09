from phi.jax.flow import *
import time


start = time.time()



def calculate_cfl(DX, DT, heat_cond_coeff):
    return DT * heat_cond_coeff / (DX**2)



reference_source = CenteredGrid(lambda x : 10* math.exp(-x**2/0.1), extrapolation.ZERO, x=N, bounds=Box(x=(-1,1)))

@math.jit_compile
def step(temperature, source,heat_cond_coeff, DT):
    T1 = diffuse.explicit(temperature, heat_cond_coeff, DT)
    T1 += source * DT
    return T1

def run_heat_simulation(source):

    N = 128
    DX = 2./N
    STEPS = 300
    DT = 1./STEPS
    heat_cond_coeff= 0.01

    print("CFL: ", calculate_cfl(DX, DT, heat_cond_coeff))

    INITIAL_NUMPY = np.asarray([1 for x in np.linspace(-1+DX/2,1-DX/2,N)])
    INITIAL = math.tensor(INITIAL_NUMPY, spatial('x') )

    temperature = CenteredGrid(INITIAL, extrapolation.BOUNDARY, x=N, bounds=Box(x=(-1,1)))
    temperatures = [temperature]
    age = 0.

    for i in range(STEPS):
        T1 = step(temperatures[-1], source,heat_cond_coeff, DT)
        temperatures.append(T1)

    return temperatures


reference_temperatures = run_heat_simulation(reference_source)
sensor_field = CenteredGrid(x = 3, bounds=Box(x=(-1,1)))

N = 128
source = CenteredGrid(1, extrapolation.BOUNDARY, x=N, bounds=Box(x=(-1,1)))

@math.jit_compile
def loss(source):
    temperatures = run_heat_simulation(source)
    loss = 0.
    
    for i in range(len(temperatures)):
        loss += field.l2_loss(reference_temperatures[i]@sensor_field- temperatures[i]@sensor_field)
    return loss,temperatures

gradient_function = math.gradient(loss)


LR = 0.01

grads=[]
for optim_step in range(100):
    (loss,temperatures), grad = gradient_function(source)
    print('Optimization step %d, loss: %f' % (optim_step,loss))
    grads.append( grad[0] )

    source = source - LR * grads[-1]

end = time.time()

print("Time elapsed: ", end - start)

temps = [t.values.numpy('x,vector') for t in reference_temperatures]

import pylab



fig = pylab.figure().gca()
pltx = np.linspace(-1,1,N)
fig.plot(pltx, reference_source.values.numpy('x,vector'), lw=2, color='red', label="source")
fig.plot(pltx, source.values.numpy('x,vector'), lw=2, color='black', label="optimized source")
pylab.legend()
pylab.show()