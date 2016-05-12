# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:49:47 2016

@author: muhaddisa
"""

# plot of samples size versus power of test for alpha =0.05 (added to
# the paper first figure)
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(True)

if __name__ == '__main__':
    figs = [1, 2, 9]

    f = 0
    if f in figs:
        fig = plt.figure(f)
        sample_size = [10, 20, 30, 40, 50, 60, 70, 80, 100]
        x_perm = [0.141, 0.27, 0.419, 0.557, 0.755, 0.862, 0.928, 0.966, 0.99]
        x_bin = [0.197, 0.394, 0.582, 0.697, 0.87, 0.913, 0.97, 0.991, 0.998]
        x_conf_I = [0.197, 0.324, 0.455, 0.656, 0.796, 0.904, 0.949, 0.977, 0.997]
        plt.plot(sample_size, x_perm, 'r', lw=2, linestyle='--',
                 label="Permutation test")
        plt.plot(sample_size, x_bin, 'g', lw=2, label="Binomial test")
        plt.plot(sample_size, x_conf_I, 'b', lw=2, linestyle='-.',
                 label="Confidence Interval test")
        plt.ylim([0, 1])
        plt.xlabel('sample size')
        plt.ylabel('Power')
        plt.title(r' Significance level $\alpha = 0.05$')
        plt.show()
        plt.legend(loc=4)
        # above plot in terms of Type II error.

    f = 1
    if f in figs:
        fig1 = plt.figure(f)
        plt1 = fig1.add_subplot(211)
        plt2 = fig1.add_subplot(212)
        sample_size = [10, 20, 30, 40, 50, 60, 70, 80, 100]
        x1_perm = [0.859, 0.73, 0.581, 0.443, 0.245, 0.138, 0.072, 0.034, 0.01]
        x1_bin = [0.803, 0.606, 0.418, 0.303, 0.13, 0.087, 0.03, 0.009, 0.002]
        x1_conf_I = [0.803, 0.676, 0.545, 0.344, 0.204, 0.096, 0.051, 0.023, 0.003]
        plt1.plot(sample_size, x1_perm, 'r', lw=2, linestyle='--', label="Perm.")
        plt1.plot(sample_size, x1_bin, 'g', lw=2, label="Bin.")
        plt1.plot(sample_size, x1_conf_I, 'b', lw=2, linestyle='-.', label="CI")
        plt.ylim([0, 1])
        # plt.xlabel('sample size')
        plt1.set_ylabel('$P(Type II)$')
        plt1.set_title(r'Significance / Confidence level : $0.05$')
        leg = plt1.legend(loc=1)
        x2_bin = [0.078, 0.086, 0.103, 0.076, 0.082, 0.057, 0.055, 0.079, 0.061]
        x2_perm = [0.059, 0.05, 0.07, 0.03, 0.03, 0.05, 0.07, 0.04,
                   0.036]  # 100 repetitions
        x2_conf_I = [0.078, 0.061, 0.045, 0.059, 0.054, 0.046, 0.026,
                     0.042, 0.041]  # 1000 repititions
        plt2.plot(sample_size, x2_perm, 'r', lw=2, linestyle='--', label="Perm.")
        plt2.plot(sample_size, x2_bin, 'g', lw=2, label="Bin.")
        plt2.plot(sample_size, x2_conf_I, 'b', lw=2, linestyle='-.', label="CI")
        plt.ylim([0, 0.3])
        plt2.set_xlabel('sample size ($N$)')
        plt2.set_ylabel('$P(Type I)$')
        # plt.title(r' Significance level $\alpha=0.05$')
        leg = plt2.legend(loc=1)
        # plt.show()
        plt.savefig("bin_ci_perm_N.pdf")

    f = 2
    if f in figs:
        # Effect of increasing dimesnion with fixed 10 d_inf (Plot
        # added-second figure)
        fig2 = plt.figure(f)
        plt1 = fig2.add_subplot(211)
        plt2 = fig2.add_subplot(212)    
        x = [10, 50, 70, 90, 200, 300]
        a1 = [0.11, 0.33, 0.45, 0.51, 0.7, 0.69]
        a2 = [0.12, 0.4, 0.58, 0.58, 0.74, 0.8]
        a3 = [0.2, 0.46, 0.6, 0.61, 0.78, 0.84]
        plt1.plot(x, a1, 'g', lw=2, label="Bin.")
        plt1.plot(x, a2, 'b', lw=2, linestyle='-.',
                  label="CI")
        plt1.plot(x, a3, 'r', lw=2, linestyle='--', label="Perm.")
        plt.ylim([0, 1])
        # plt.xlabel(' Dimesnion (d)')
        plt1.set_ylabel('$P(Type II)$')
        plt1.set_title(r'$d_{inf}=10$, Significance / Confidence level = $0.05$')
        leg = plt1.legend(loc=4)

        xx = [10, 50, 90, 200, 300]
        aa1 = [0.06, 0.07, 0.1, 0.13, 0.15]
        aa2 = [0.03, 0.04, 0.07, 0.05, 0.11]
        aa3 = [0.02, 0.05, 0.06, 0.1, 0.11]
        plt2.plot(xx, aa1, 'g', lw=2, label="Bin.")
        plt2.plot(xx, aa3, 'b', lw=2, linestyle='-.',
                  label="CI")
        plt2.plot(xx, aa2, 'r', lw=2, linestyle='--', label="Perm.")

        plt.ylim([0, 0.2])
        plt2.set_xlabel(' Dimension ($d$)')
        plt2.set_ylabel('$P(Type I)$')
        leg = plt2.legend(loc=2)
        # plt.show()
        plt.savefig("bin_ci_perm_d.pdf")

    f = 3
    if f in figs:
        ###########################################################################
        # plot of samples size versus power of test for alpha =0.01 (not added)
        # plt.figure()
        # x1_perm = [0.07, 0.14, 0.28, 0.35, 0.46, 0.7, 0.82, 0.85, 0.98, 0.97]
        # x1_bin = [0.08, 0.17, 0.36, 0.52, 0.58, 0.75, 0.91, 0.87, 0.97, 1]
        # x1_conf_I = [0.06, 0.13, 0.3, 0.38, 0.44, 0.75, 0.86, 0.87, 0.97, 0.99]
        # plt.plot(sample_size, x1_perm, 'r', label="perm_test")
        # plt.plot(sample_size, x1_bin, 'g', label="bin_test")
        # plt.plot(sample_size, x1_conf_I, 'b', label="Conf_I test")
        # plt.ylim([0, 1])
        # plt.xlabel('Sample size')
        # plt.ylabel('Power')
        # plt.title(r'$\alpha=0.01$')
        # plt.show()
        # plt.legend(loc=4)

        # plot of effect size(d) vs power depending on no. of samples (not added)
        fig3 = plt.figure(f)
        plt1 = fig3.add_subplot(121)
        plt2 = fig3.add_subplot(122)
        x_axis = np.arange(0, 1.2, 0.2)
        # for n=10, 20, 30, 40, 50, 100
        y1_n1 = [0.08, 0.11, 0.15, 0.3, 0.53, 0.75]
        y1_n2 = [0.13, 0.14, 0.23, 0.69, 0.94, 0.99]
        y1_n3 = [0.1, 0.17, 0.51, 0.85, 1, 1]
        y1_n4 = [0.06, 0.09, 0.47, 0.95, 1, 1]
        y1_n5 = [0.1, 0.16, 0.51, 0.97, 1, 1]
        y1_n6 = [0.08, 0.27, 0.96, 1, 1, 1]
        plt1.plot(x_axis, y1_n1, 'r', lw=2, linestyle='-', label="n=10")
        plt1.plot(x_axis, y1_n2, lw=2, linestyle='-.', label="n=20")
        plt1.plot(x_axis, y1_n3, lw=2, linestyle=':', label="n=30")
        plt1.plot(x_axis, y1_n4, lw=2, linestyle='--', label="n=40")
        plt1.plot(x_axis, y1_n5, 'g', lw=2, linestyle='-', label="n=50")
        plt1.plot(x_axis, y1_n6, 'b', lw=2, linestyle='-', label="n=100")
        plt.ylim([0, 1])
        plt1.set_xlabel(r'Effect Size ($\delta$)')
        plt1.set_ylabel('power')
        plt1.set_title(r' Binomial test $\alpha=0.05$')
        leg = plt1.legend(loc=4)

        y2_n1 = [0.02, 0.04, 0.06, 0.12, 0.26, 0.56]
        y2_n2 = [0.04, 0.05, 0.10, 0.39, 0.79, 0.97]
        y2_n3 = [0.01, 0.01, 0.16, 0.58, 0.94, 1]
        y2_n4 = [0, 0.03, 0.22, 0.82, 1, 1]
        y2_n5 = [0.03, 0.06, 0.26, 0.89, 1, 1]
        y2_n6 = [0.03, 0.11, 0.88, 1, 1, 1]
        plt2.plot(x_axis, y2_n1, 'r', lw=2, linestyle='-', label="n=10")
        plt2.plot(x_axis, y2_n2, lw=2, linestyle='-.', label="n=20")
        plt2.plot(x_axis, y2_n3, lw=2, linestyle=':', label="n=30")
        plt2.plot(x_axis, y2_n4, lw=2, linestyle='--', label="n=40")
        plt2.plot(x_axis, y2_n5, 'g', lw=2, linestyle='-', label="n=50")
        plt2.plot(x_axis, y2_n6, 'b', lw=2, linestyle='-', label="n=100")
        plt.ylim([0, 1])
        plt2.set_xlabel(r'Effect Size ($\delta$)')
        plt2.set_ylabel('power')
        plt2.set_title(r' Binomial test $\alpha=0.01$')
        leg = plt2.legend(loc=4)
        # plt.show()

        # CV folds vs Type I and Type II error for Binomial test

    f = 4
    if f in figs:
        fig4 = plt.figure(f)
        plt1 = fig4.add_subplot(111)
        #plt2 = fig4.add_subplot(212)
        xaxis = [2, 3, 5, 7, 10, 15, 17]
        yaxis_2 = [0.69, 0.60, 0.52, 0.47, 0.48, 0.46, 0.50]
        yaxis_1 = [0.03, 0.07, 0.13, 0.13, 0.13, 0.14, 0.16]
        plt1.plot(xaxis, yaxis_1, 'r', lw=2, linestyle='-', label=" Type I ")
        plt1.plot(xaxis, yaxis_2, 'b', lw=2, linestyle='--',label=" Type II ")
        plt.ylim([0, 1])
        plt.xlim([2, 17])
        leg = plt1.legend(loc=1)
        plt1.set_xlabel('No. of CV folds')
        plt1.set_ylabel('Error Response')
        plt1.set_title('Binomial Test')
        #plt.show()

    f = 5
    if f in figs:
        # bar plot of Type I error using diff classifiers (not added)
        fig5 = plt.figure(f)
        plt1 = fig5.add_subplot(211)
        plt2 = fig5.add_subplot(212)
        n = [1, 2, 3, 4]
        type_1_error = [0.13, 0.05, 0.09, 0.12]
        LABELS = ["SVM (linear)", "SVM(RBF)", "LR(L1)", "LR(L2)"]
        plt1.bar(n, type_1_error, align='center')
        plt1.set_xticks(n, LABELS)
        plt1.set_ylabel('Type I error')
        plt1.set_title(r'binomial test, $\alpha=0.05$')
        # plt.show()
        n = [1, 2, 3, 4]
        type_1_error = [0.04, 0.02, 0.04, 0.04]
        LABELS = ["SVM (linear)", "SVM(RBF)", "LR(L1)", "LR(L2)"]
        plt2.bar(n, type_1_error, align='center')
        plt2.set_xticks(n, LABELS)
        plt2.set_ylabel('Type I error')
        plt2.set_title(r'binomial test, $\alpha=0.01$')
        # plt.show()

    f = 6
    if f in figs:
        # plot of dimension (d) vs Type I error (not added)
        fig6 = plt.figure(f)
        plt1 = fig6.add_subplot(111)
        dim = [30, 50, 100, 200, 300]
        err_1 = [0.08, 0.07, 0.08, 0.13, 0.15]
        err_2 = [0.01, 0.03, 0.03, 0.04, 0.06]
        plt1.plot(dim, err_1, 'r', label=" significance level=0.05")
        plt1.plot(dim, err_2, 'b', label=" significance level=0.01")
        leg = plt1.legend(loc=1)
        plt1.set_xlabel('dimension (d)')
        plt1.set_ylabel('Type I error')
        plt1.set_title('Binomial Test')
        # plt.show()

    f = 7
    if f in figs:
        # plot of Power vs covariance of class B (not added)
        fig7 = plt.figure(f)
        plt1 = fig7.add_subplot(211)
        n = [1, 2, 3, 4]
        power_1 = [0.394, 0.31, 0.296, 0.267]
        power_2 = [0.179, 0.143, 0.112, 0.079]
        LABELS = ["COV_B=1", "COV_B=1.5", "COV_B=1.75", "COV_B=2"]
        plt1.bar(n, power_1, align='center')
        plt1.set_xticks(n, LABELS)
        plt1.set_ylabel('Power')
        plt1.set_title(r'binomial test, $\alpha=0.05$')
        # plt.show()

        plt2 = fig7.add_subplot(212)
        n = [1, 2, 3, 4]
        LABELS = ["COV_B=1", "COV_B=1.5", "COV_B=1.75", "COV_B=2"]
        plt2.bar(n, power_2, align='center')
        plt2.set_xticks(n, LABELS)
        plt2.set_ylabel('Power')
        plt2.set_title(r'binomial test, $\alpha=0.01$')
        # plt.show()

    f = 8
    if f in figs:
        # plot of power vs d_inf ()
        fig8 = plt.figure(f)
        plt1 = fig8.add_subplot(121)
        plt2 = fig8.add_subplot(122)

        x = [10, 20, 30, 40, 50]
        y1 = [0.394, 0.732, 0.923, 0.986, 1]
        y2 = [0.373, 0.753, 0.958, 0.999, 1]

        plt1.plot(x, y1, 'r', label="eye(d)")
        plt1.plot(x, y2, 'b', label="U(0, 1)")
        leg = plt1.legend(loc=1)
        plt1.set_xlabel('d_inf')
        plt1.set_ylabel('Power')
        plt1.set_title(r'binomial test, $\alpha=0.05$')

        yy1 = [0.179, 0.51, 0.795, 0.953, 0.988]
        yy2 = [0.191, 0.524, 0.855, 0.989, 0.999]
        plt2.plot(x, yy1, 'r', label="eye(d)")
        plt2.plot(x, yy2, 'b', label="U(0, 1)")
        leg = plt2.legend(loc=1)
        plt2.set_xlabel('d_inf')
        plt2.set_ylabel('Power')
        plt2.set_title(r'binomial test, $\alpha=0.01$')
        plt.show()
    f = 9
    if f in figs:
        fig9 = plt.figure(f)
        plt1 = fig9.add_subplot(211)
        plt2 = fig9.add_subplot(212)
        xaxis = [2, 3, 5, 7, 10, 13, 15, 17]
        CI_II = [0.77, 0.66, 0.63, 0.57, 0.56, 0.52, 0.6, 0.57]
        perm_II = [0.81, 0.76, 0.68, 0.64, 0.65, 0.65, 0.63, 0.64]
        bin_II= [0.69, 0.6, 0.52, 0.47, 0.48, 0.44, 0.46, 0.5]
        plt1.plot(xaxis, perm_II, 'r', lw=2, linestyle='--', label="Perm.")
        plt1.plot(xaxis, bin_II, 'g', lw=2, label="Bin.")
        plt1.plot(xaxis, CI_II, 'b', lw=2, linestyle='-.', label="CI")
        #plt.ylim([0, 1])
        plt1.set_ylim([0.0,1])
        plt1.set_xlim([2,17])
        # plt.xlabel('sample size')
        plt1.set_ylabel('$P(Type II)$')
        plt1.set_title(r'Significance / Confidence level : $0.05$')
        leg = plt1.legend(loc=3)
        CI_I = [0.01, 0.06, 0.1, 0.1, .06, 0.11, 0.1, 0.1]
        perm_I = [0.02, 0.03, 0.05, 0.04, 0.09, 0.08, 0.07, 0.07]
        bin_I= [0.03, 0.07, 0.13, 0.13, 0.12, 0.12, 0.14, 0.16]
        plt2.plot(xaxis, perm_I, 'r', lw=2, linestyle='--', label="Perm.")
        plt2.plot(xaxis, bin_I, 'g', lw=2, label="Bin.")
        plt2.plot(xaxis, CI_I, 'b', lw=2, linestyle='-.', label="CI")
        #plt.ylim([0, 0.5])
        plt2.set_ylim([0,0.3])
        plt2.set_xlim([2,17])
        plt2.set_xlabel('$k$')
        plt2.set_ylabel('$P(Type I)$')
        leg = plt2.legend(loc=2)
        plt.show()
        plt.savefig("bin_ci_perm_cv.pdf")
