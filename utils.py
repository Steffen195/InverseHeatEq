import matplotlib.pyplot as plt

def calculate_cfl(DX, DT, heat_cond_coeff):
    return DT * heat_cond_coeff / (DX**2)


def plot_heatmap(trj, Lt,Lx):
    plt.figure(figsize=(20, 5))
    plt.imshow(
        trj.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
    )
    plt.colorbar()
    plt.xlabel("time")
    plt.ylabel("space")
    plt.show()

def plot_ref_and_test_heatmap(trj_ref,trj_test,Lt,Lx):
    
    ax, fig = plt.subplots(1,2,figsize=(20, 5))
    fig[0].imshow(
        trj_ref.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
    )
    fig[0].set_title("Reference solution")
    fig[0].set_xlabel("time")
    fig[0].set_ylabel("space")
    fig[1].imshow(
        trj_test.T,
        cmap="hot",
        aspect="auto",
        origin="lower",
        extent=(0, Lt, -Lx/2, Lx/2),
    )
    fig[1].set_title("Test solution")
    fig[1].set_xlabel("time")
    fig[1].set_ylabel("space")
    plt.show()
    
def plot_diffusivity(ref_diffusivity, test_diffusivity, x, sensor_positions):
    plt.figure(figsize=(20, 5))
    plt.plot(x, ref_diffusivity, label="reference")
    plt.plot(x, test_diffusivity, label="test")
    plt.legend()
    plt.xlabel("space")
    plt.ylabel("diffusivity")
    plt.vlines(sensor_positions, ymin=0, ymax=1, color="black", linestyles="dashed")
    plt.show()