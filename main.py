# This is a sample Python script.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pymcr.mcr import McrAR

from pymcr.constraints import ConstraintNonneg, ConstraintNorm


def mcrDemo(D, ncomponents):

    mcrar = McrAR(max_iter=100, st_regr='NNLS', c_regr='NNLS', st_constraints=[ConstraintNorm()], c_constraints=[ConstraintNonneg()], tol_increase=100)

    initialProfiles = np.random.rand(np.size(D, 0), ncomponents)

    mcrar.fit(D, C=initialProfiles, verbose=True)

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(D)
    ax[0, 0].title.set_text("Raw GC-MS Data")
    ax[0, 0].set_ylabel("Counts")
    ax[0, 0].set_xlabel("Acquisitions")

    ax[0, 1].plot(mcrar.C_opt_)
    ax[0, 1].set_ylabel("Score Intensity")
    ax[0, 1].set_xlabel("Acquisitions")
    ax[0, 1].title.set_text("Elution Profiles (C)")

    ax[1, 0].plot(np.arange(35, 601).T, mcrar.ST_opt_.T)
    ax[1, 0].set_ylabel("Intensity, Normalised")
    ax[1, 0].set_xlabel("m/z")
    ax[1, 0].title.set_text("Deconvolved Mass Spectra (S.T)")
    ax[1, 0].axis(xmin=35, xmax=250)

    ax[1, 1].plot(np.log(mcrar.err[::2] / np.linalg.norm(D)))
    ax[1, 1].set_ylabel("log10 MSE")
    ax[1, 1].set_xlabel("Iterations")
    ax[1, 1].title.set_text("Minimisation of Error")

    plt.show()


if __name__ == '__main__':
    D = pd.read_csv("gcms1.csv", sep=",").to_numpy()
    # plt.plot(D)
    # plt.show()
    mcrDemo(D, 4)


