import matplotlib.pyplot as plt



fig, axes = plt.subplots(3,3, figsize = (30, 20))
for (desc, group), ax in zip(Energy_sources, axes.flatten()):
    group.plot(y = 'Value',ax = ax, title = desc, fontsize = 25)
    ax.set_xlabel('时间')
    ax.set_ylabel(u'碳排放量（百万公吨）')
    ax.xaxis.label.set_size(23)
    ax.yaxis.label.set_size(23)
