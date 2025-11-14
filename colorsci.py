
import numpy as np
import matplotlib.pyplot as plt
import colour
from colour.plotting import plot_chromaticity_diagram_CIE1931,plot_chromaticity_diagram_CIE1960UCS,plot_sds_in_chromaticity_diagram_CIE1976UCS,plot_chromaticity_diagram_CIE1976UCS
from colour import SpectralDistribution
# 随机点
np.random.seed(42)
x_rand = np.random.uniform(0.25, 0.45, 100)
y_rand = np.random.uniform(0.25, 0.45, 100)

# plt.figure(figsize=(8, 8))
plot_chromaticity_diagram_CIE1931(show_diagram_colours=True,show_spectral_locus=True)
plot_chromaticity_diagram_CIE1960UCS(show_diagram_colours=True,show_spectral_locus=True)
plot_chromaticity_diagram_CIE1976UCS(show_diagram_colours=True,show_spectral_locus=True)
# plt.scatter(x_rand, y_rand, c='red', s=20, label="Random xy")
# plt.legend()
# plt.title("CIE 1931 Chromaticity Diagram with Random xy Points")
# plt.show()
