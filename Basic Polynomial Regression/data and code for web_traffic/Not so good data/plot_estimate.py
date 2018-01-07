def func(n):
    import scipy as sp
    import matplotlib.pyplot as plt
    #assuming data is per hour
    data = sp.genfromtxt("web_traffic.tsv",delimiter="\t")
    x  = data[: , 0]
    y = data[: , 1]
    x = x[~sp.isnan(y)]
    y=y[~sp.isnan(y)]
    plt.scatter(x,y,marker='+',color='g')
    plt.title("Web traffic")
    plt.xlabel("time")
    plt.ylabel("hits/hr")
    plt.xticks([w*7*24 for w in range(10)],['week %i' %w for w in range(10)])
    fpn = sp.polyfit(x,y,n)
    fn = sp.poly1d(fpn)
    fx = sp.linspace(0,x[-1],1000) #generate x values for plotting
    plt.plot(fx,fn(fx),color='k',linewidth = 3)
    plt.legend(["d=%i"%fn.order],loc = "upper center")
    plt.grid()
    plt.show()
    return fn,x,y

