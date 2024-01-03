import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.size"] = 20

def calculate_cfl(DX, DT, heat_cond_coeff):
    return DT * heat_cond_coeff / (DX**2)


def plot_heatmap(trj, Lt,Lx):
    plt.figure(figsize=(22, 5))
    plt.imshow(
        trj.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
    )
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Space")
    plt.title("Temperature")
    plt.show()

    

def plot_ref_and_test_heatmap(trj_ref,trj_test,Lt,Lx,x,test_case):
    
    Tmax = np.max([trj_ref,trj_test])
    Tmin = np.min([trj_ref,trj_test])

    fig, axes = plt.subplots(1,2,figsize=(22, 5),layout = "compressed")
    #fig.suptitle(f"Temperature for {test_case} Source")
    axes[0].imshow(
        trj_ref.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
        vmin = Tmin,
        vmax = Tmax

    )
    axes[0].set_title("Reference Solution")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Space")
    im = axes[1].imshow(
        trj_test.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
        vmin = Tmin,
        vmax = Tmax
    )
    axes[1].set_title("Test Solution")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Space")
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
    
    return fig



def plot_ref_initial_and_test_heatmap(trj_ref,trj_ini,trj_test,Lt,Lx,x,test_case):
    
    Tmax = np.max([trj_ref,trj_ini,trj_test])
    Tmin = np.min([trj_ref,trj_ini,trj_test])
    fig, axes = plt.subplots(1,3,figsize=(22, 5),layout = "compressed")
    #fig.suptitle(f"Heat source for {test_case} Source")
    axes[0].imshow(
        trj_ref.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
        vmin = Tmin,
        vmax = Tmax
    )
    axes[0].set_title("Reference Solution")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Space")
    axes[1].imshow(
        trj_ini.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
        vmin = Tmin,
        vmax = Tmax,
    )
    axes[1].set_title("Initial guess")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Space")

    im = axes[2].imshow(
        trj_test.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
        vmin = Tmin,
        vmax = Tmax,
    )
    axes[2].set_title("Test Solution")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Space")

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

    return fig


def plot_diffusivity(ref_diffusivity, initial_diffusivity, test_diffusivity, x, sensor_positions):
    
    ymax = 1.1*np.max([np.max(ref_diffusivity),np.max(test_diffusivity),np.max(initial_diffusivity)])
    ymin = 0.9*np.min([np.min(ref_diffusivity),np.min(test_diffusivity),np.min(initial_diffusivity)])
    plt.figure(figsize=(20, 5))
    plt.plot(x, ref_diffusivity, label="Reference")
    plt.plot(x, initial_diffusivity,label="Initial")
    plt.plot(x, test_diffusivity, label="Test")
    plt.legend()
    plt.xlabel("Space")
    plt.ylabel("Diffusivity")
    plt.ylim([ymin,ymax])
    plt.xlim([np.min(x),np.max(x)])
    #plt.vlines(sensor_positions, ymin=ymin, ymax=ymax, color="black", linestyles="dashed")
    plt.show()

