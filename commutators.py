import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax
jax.config.update('jax_platform_name', 'cpu')

platform = jax.lib.xla_bridge.get_backend().platform.casefold()
#print("Platform: ", platform)

NUM_PROC = 56
FOLDER = "./output/"
#print(jax.device_count(backend="cpu"))

import jax.numpy as np

#import multiprocessing as mp
from quadax import quadgk

from functools import partial

def integrand(r, x, t, a):
    f = np.exp(-1/2 * r**2)
    return r**2 * f**2 * np.sinc(r*x / np.pi) * np.sinc(t * f * np.sqrt(r**2 + a**2) / np.pi)

@partial(jax.jit, static_argnames=['a'])
def comm(x, t, a):
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    I, _ = quadgk(lambda r: integrand(r, x, t, a), [0, +np.inf])  
    
    return abs(1/(2 * np.pi**2) * t * I)**2

def calc(lim_x, lim_t, step, alpha, hat_coords=False):
    xs = np.linspace(-lim_x, lim_x, step)
    ts = np.linspace(-lim_t, lim_t, step)
    
    if hat_coords:
        xs = xs / alpha
        ts = ts / alpha
    
    # manual parallel processing
    #arguments = [(x, t, alpha) for t in ts for x in xs]
    #mp_context = mp.get_context("spawn")
    #with mp_context.Pool(processes=NUM_PROC) as pool:
    #    z = pool.starmap(comm, arguments)
    #z = np.array(z).reshape(step, step)
    
    # vectorization with JAX
    vcomm = jax.vmap(jax.vmap(lambda x, t: comm(x, t, alpha), in_axes=(0, None)), in_axes=(None, 0))
    z = vcomm(xs, ts)
    
    xx, tt = np.meshgrid(xs, ts)
    
    return xx, tt, z

import matplotlib.pyplot as plt

def plot(xx, tt, z, sigma, mass, label=None, log=False, show=True, hat_coords=False):
    ax = plt.gca()
    ax.axis('equal')
    ax.set_box_aspect(1)
    
    alpha = sigma*mass
    coord_label = "/\sigma"
    if hat_coords:
        xx = alpha * xx
        tt = alpha * tt
        coord_label = "\cdot m"
    
    # heat_r, gist_heat_r, ...
    cnt = plt.pcolormesh(xx, tt, z, cmap='gist_heat_r', antialiased=False, shading='auto')
    #plt.contourf(xx, tt, z)
    #plt.imshow(z)
    
    plt.colorbar()
    plt.xlabel(f"Space Coordinate $x{coord_label}$")
    plt.ylabel(f"Time Coordinate $t{coord_label}$")
    plt.title("$" + (r"\log" if log else "") + "|\sigma^2 [\phi(t'',0,0,x''), \phi(0,0,0,0)]|^2$, "
                  + r"$\alpha" + f" = {alpha}$")
    
    if label:
        plt.savefig(FOLDER + label + ("-log" if log else "") + ".png")
    
    if show:
        plt.show()
    
def run1(lim_x, lim_t, step, sigma, mass, label=None, log=False, show=True, hat_coords=False):
    alpha = sigma * mass
    
    xx, tt, z = calc(lim_x, lim_t, step, alpha, hat_coords=hat_coords)
    
    if not label:
        label = f"numerical-commutator-m{mass}-s{sigma}-l{lim_x}-{lim_t}"
    
    plot(xx, tt, z**(1/4), sigma, mass, label=label, show=show, hat_coords=hat_coords)
    
    if log:
        plot(xx, tt, np.log(z), sigma, mass, label=label, log=True, show=show)

def run2(alpha):
    """ To be parallelized. """
    run1(20, 20, 400, alpha, 1, log=False, show=False, hat_coords=True)

