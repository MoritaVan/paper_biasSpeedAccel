
def pause():
    x = input("Press the <ENTER> key to continue...")
    print(x)


def signaltonoise(a, axis=0, ddof=0):
    import numpy as np
    
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)



def longestNanRun(sequence):
    import numpy as np

    nan_run = np.diff(np.concatenate(([-1], np.where(~np.isnan(sequence))[0], [len(sequence)])))
    nan_seq = np.where(nan_run>1)[0]

    nan_run = nan_run[nan_seq]

    seqs = np.split(nan_seq, np.where(np.diff(nan_seq) != 1)[0]+1)
    final_nan_run = []
    for seq in seqs:
        idx = np.searchsorted(nan_seq,seq)
        final_nan_run.append(sum(nan_run[idx]))

    return max(final_nan_run, default=0)

# got nan_helper from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    import numpy as np

    return np.isnan(y), lambda z: z.nonzero()[0]


def plotFig(trial_idx, tg_vel, time_x, vel_x, equ_x, start_a_x, lat_x, ax=None, show=False) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    box_x = (time_x > start_a_x) & (time_x < lat_x)
    
    if ax == None:
        f = plt.figure(figsize=(7,4))
    
    plt.title('Trial %d' % trial_idx)
    plt.hlines(y = 0, xmin=time_x[0], xmax=0, linewidth = 1, linestyle = '--', color = 'k')
    plt.hlines(y = tg_vel, xmin=0, xmax=500, linewidth = 1, linestyle = '--', color = 'k')
    plt.vlines(x = 0, ymin=0, ymax=tg_vel, linewidth = 1, linestyle = '--', color = 'k')
    plt.plot(time_x, vel_x, color = np.array([5, 94, 255])/255)
    plt.plot(time_x, equ_x, color = np.array([255, 35, 0])/255)
    plt.fill_between(time_x, -40, 40, where=box_x, color='gray', alpha=0.1, interpolate=True)
    plt.text(start_a_x, 17, 'anticip. onset', ha='right', va='top', rotation=90)
    plt.text(lat_x, 17, 'anticip. offset', ha='right', va='top', rotation=90)
    plt.ylim(-2,18)
    plt.xlabel('Time (ms)')
    plt.ylabel('Velocity (deg/s) x-axis')

    if show:
        plt.show()

def plotFig2(trial_idx, tg_time, tg_vel, tg_dir_v, tg_dir_h, time_y, vel_y, equ_y, start_a_y, lat_y, time_x, vel_x, equ_x, start_a_x, lat_x, ax=None, show=False) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    
    dir_v = 1 if tg_dir_v=='U' else -1
    dir_h = 1 if tg_dir_h=='R' else -1

    box_x = (time_x > start_a_x) & (time_x < lat_x)
    box_y = (time_y > start_a_y) & (time_y < lat_y)

    if ax == None:
        f = plt.figure(figsize=(15,4))
        
    plt.suptitle('Trial %d' % trial_idx)
    plt.subplot(1,2,1)
    plt.plot(tg_time, tg_vel*dir_h, linewidth = 1, linestyle = '--', color = '0.5')
    plt.plot(time_x, vel_x, color = np.array([5, 94, 255])/255)
    plt.plot(time_x, equ_x, color = np.array([255, 35, 0])/255)
    plt.axvline(x = 0, linewidth = 1, linestyle = '--', color = 'k')
    plt.fill_between(time_x, -40, 40, where=box_x, color='gray', alpha=0.1, interpolate=True)
    plt.text(start_a_x, 30, 'anticip. onset', ha='right', va='top', rotation=90)
    plt.text(lat_x, 30, 'anticip. offset', ha='right', va='top', rotation=90)
    # plt.ylim(-35,35)
    plt.ylim(-5,32)
    plt.xlabel('Time (ms)')
    plt.ylabel('Velocity (deg/s) x axis')
    plt.subplot(1,2,2)
    plt.plot(tg_time, tg_vel*dir_v, linewidth = 1, linestyle = '--', color = '0.5')
    plt.plot(time_y, vel_y, color = np.array([5, 94, 255])/255)
    plt.plot(time_y, equ_y, color = np.array([255, 35, 0])/255)
    plt.axvline(x = 0, linewidth = 1, linestyle = '--', color = 'k')
    plt.fill_between(time_y, -40, 40, where=box_y, color='gray', alpha=0.1, interpolate=True)
    plt.text(start_a_y, 30, 'anticip. onset', ha='right', va='top', rotation=90)
    plt.text(lat_y, 30, 'anticip. offset', ha='right', va='top', rotation=90)
    # plt.ylim(-35,35)
    plt.ylim(-5,32)
    plt.xlabel('Time (ms)')
    plt.ylabel('Velocity (deg/s) y axis')

    if show:
        plt.show()


def closest(lst, K): 
    import numpy as np

    lst = np.asarray(lst) 
    idx = (np.abs(lst - K)).argmin() 
    return lst[idx], idx


def old_plotBoxDispersion(data, by:str, between:str, groups=None, groupsNames=None, ax=None, jitter=.125, scatterSize:int=.5, showfliers:bool=True, alpha:int=10, showKde:bool=True, showBox:bool=True, xticks=None, cmapName:str='winter') -> None:
    '''
    ----------------------------------
    Created by Cristiano Azarias, 2020
    ----------------------------------
    Adapted by Vanessa
    ----------------------------------
    data: data to plot
    by: (list of) variable(s) to group data
    between: (list of) variable(s) to return for grouped data
    alpha: integer to scale the amplitude of the kde dist

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    if ax is None: # Check wether an axes was assigned or not
        fig, ax = plt.subplots() # Create figure
        fig.set_size_inches((6,6)) # Set a size
    
    group_by = data.groupby(by)[between] # Create group generator
    if groups is None:
        n_groups = len(group_by.groups)
    else:
        n_groups = len(groups)

    cmap = plt.get_cmap(cmapName) #timing_cmap() # Load colormap
    colors = cmap(np.linspace(.15, .65, n_groups)) # Set colors based on the group length
    colors50 = np.copy(colors) # Colors copy
    colors50[:,-1] = .7 # Decrease opacity
    colors80 = np.copy(colors)
    colors80[:,-1] = .9 # Decrease opacity

    if xticks is not None:
        pos = xticks
    else:
        pos = np.arange(n_groups)+1
    
    if groups is None:
        for idx, group in enumerate(group_by):
            if showBox:
                bplot = ax.boxplot(group[1], positions = [pos[idx]], patch_artist=True, zorder=2, showfliers=showfliers, widths=3) # Plot boxplot
            kde = stats.gaussian_kde(group[1]) # Fit gaussian kde
            if xticks is not None:
                x = np.linspace(min(xticks), max(xticks), 1000)
            else:
                x = np.linspace(min(group[1]), max(group[1]), 1000) # Assing x's
            n = len(group[1]) # Assign the number of items per group
            amp = alpha * n / len(data) # Set the amplitude based on the ratio of group size and total items
            disp = jitter * np.abs(np.random.randn(n)) # Assign dispersion ratio
            ax.scatter(pos[idx]*np.ones(n)+disp, group[1], s=scatterSize, facecolor=colors80[idx], zorder=1) # Plot all data on the right side of boxplot
            if showKde:
                ax.fill_betweenx(x, pos[idx]-kde(x)*amp, pos[idx], facecolor=colors80[idx-1], zorder=1) # Plot the kde curve on the left side of boxplot
            if showBox:
                for patch in bplot['boxes']: 
                    patch.set_facecolor((0,0,0,0)) # Set boxplot to transparent
    #                 patch.set_edgecolor(colors[group[0]-1]) # Set boxplot edgecolor to black
                    patch.set_edgecolor((0,0,0,1)) # Set boxplot edgecolor to black

                for patch in bplot['medians']: 
                    patch.set_color('gold') # Set boxplot median to dark yellow
                    patch.set_linewidth(2) # Set boxplot median line width to 2
                   
        # print(group_by.groups)
        # ax.set_xticklabels(group_by.groups.keys())
        plt.ylabel(between)
    else:
        for g in groups:
            try: 
                grouptmp = group_by.get_group(g)
                group = grouptmp.dropna()
                idx   = groups.index(g) 

                if showBox:
                    bplot = plt.boxplot(group, positions = [pos[idx]], patch_artist=True, zorder=2, showfliers=showfliers, widths=3) # Plot boxplot
                kde = stats.gaussian_kde(group) # Fit gaussian kde
                x = np.linspace(min(group), max(group), 1000) # Assign x's
                n = len(group) # Assign the number of items per group
                amp = alpha * n / len(data) # Set the amplitude based on the ratio of group size and total items
                disp = jitter * np.abs(np.random.randn(n)) # Assign dispersion ratio
                plt.scatter(pos[idx]*np.ones(n)+disp, group, s=scatterSize, facecolor=colors80[pos[idx]-1], zorder=1) # Plot all data on the right side of boxplot
                if showKde:
                    plt.fill_betweenx(x, pos[idx]-kde(x)*amp, pos[idx], facecolor=colors80[pos[idx]-1], zorder=1) # Plot the kde curve on the left side of boxplot
                if showBox:
                    for patch in bplot['boxes']: 
                        patch.set_facecolor((0,0,0,0)) # Set boxplot to transparent
    #                     patch.set_edgecolor(colors[pos[idx]-1]) # Set boxplot edgecolor to black
                        patch.set_edgecolor((0,0,0,1)) # Set boxplot edgecolor to black

                    for patch in bplot['medians']: 
                        patch.set_color('gold') # Set boxplot median to dark yellow
                        patch.set_linewidth(2) # Set boxplot median line width to 2
            
            except: continue

        plt.xticks(pos, groupsNames)
        # plt.xlim(pos[0]-1, pos[-1]+1)
        plt.ylabel(between)


def plotBoxDispersion(data, by:str, between:str, groups=None, groupsNames=None, ax=None, 
                      jitter=.125, scatterSize:int=.5, boxWidth:float=5., showfliers:bool=True, alpha:int=10, 
                      showKde:bool=True, showBox:bool=True, 
                      xticks=None, cmapName:str='winter', cmapAlpha=.5) -> None:
    '''
    ----------------------------------
    Created by Cristiano Azarias, 2020
    ----------------------------------
    Adapted by Vanessa
    ----------------------------------
    data: data to plot
    by: (list of) variable(s) to group data
    between: (list of) variable(s) to return for grouped data
    alpha: integer to scale the amplitude of the kde dist

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    if ax is None: # Check wether an axes was assigned or not
        fig, ax = plt.subplots() # Create figure
        fig.set_size_inches((6,6)) # Set a size
    
    group_by = data.groupby(by)[between] # Create group generator
    if groups is None:
        n_groups = len(group_by.groups)
    else:
        n_groups = len(groups)

    cmap = plt.get_cmap(cmapName) #timing_cmap() # Load colormap
    colors = cmap(np.linspace(.3, .6, n_groups)) # Set colors based on the group length
    colors50 = np.copy(colors) # Colors copy
    colors50[:,-1] = .7 # Decrease opacity
    colors80 = np.copy(colors)
    colors80[:,-1] = .9 # Decrease opacity
    colorsAlpha = np.copy(colors)
    colorsAlpha[:,-1] = cmapAlpha # Decrease opacity

    if xticks is not None:
        pos = xticks
    else:
        pos = np.arange(n_groups)+1
    
    if groups is None:
        for idx, group in enumerate(group_by):
            if showBox:
                bplot = ax.boxplot(group[1], positions = [pos[idx]], patch_artist=True, zorder=2, showfliers=showfliers, widths=boxWidth) # Plot boxplot
            kde = stats.gaussian_kde(group[1]) # Fit gaussian kde
            if xticks is not None:
                x = np.linspace(min(xticks), max(xticks), 1000)
            else:
                x = np.linspace(min(group[1]), max(group[1]), 1000) # Assing x's
            n = len(group[1]) # Assign the number of items per group
            amp = alpha * n / len(data) # Set the amplitude based on the ratio of group size and total items
            disp = jitter * np.abs(np.random.randn(n)) # Assign dispersion ratio
            ax.scatter(pos[idx]*np.ones(n)+disp, group[1], s=scatterSize, facecolor=colorsAlpha[idx], zorder=1) # Plot all data on the right side of boxplot
            if showKde:
                ax.fill_betweenx(x, group[0]-kde(x)*amp, group[0], facecolor=colorsAlpha[idx-1], zorder=1) # Plot the kde curve on the left side of boxplot
            if showBox:
                for patch in bplot['boxes']: 
                    patch.set_facecolor((0,0,0,0)) # Set boxplot to transparent
    #                 patch.set_edgecolor(colors[group[0]-1]) # Set boxplot edgecolor to black
                    patch.set_edgecolor((0,0,0,1)) # Set boxplot edgecolor to black

                for patch in bplot['medians']: 
                    patch.set_color('gold') # Set boxplot median to dark yellow
                    patch.set_linewidth(2) # Set boxplot median line width to 2
                   
        # print(group_by.groups)
        # ax.set_xticklabels(group_by.groups.keys())
        plt.ylabel(between)
    else:
        for g in groups:
            try: 
                grouptmp = group_by.get_group(g)
                group = grouptmp.dropna()
                idx   = groups.index(g) 

                if showBox:
                    bplot = plt.boxplot(group, positions = [pos[idx]], patch_artist=True, zorder=2, showfliers=showfliers) # Plot boxplot
                kde = stats.gaussian_kde(group) # Fit gaussian kde
                x = np.linspace(min(group), max(group), 1000) # Assign x's
                n = len(group) # Assign the number of items per group
                amp = alpha * n / len(data) # Set the amplitude based on the ratio of group size and total items
                disp = jitter * np.abs(np.random.randn(n)) # Assign dispersion ratio
                plt.scatter(pos[idx]*np.ones(n)+disp, group, s=scatterSize, facecolor=colorsAlpha[pos[idx]-1], zorder=1) # Plot all data on the right side of boxplot
                if showKde:
                    plt.fill_betweenx(x, pos[idx]-kde(x)*amp, pos[idx], facecolor=colorsAlpha[pos[idx]-1], zorder=1) # Plot the kde curve on the left side of boxplot
                if showBox:
                    for patch in bplot['boxes']: 
                        patch.set_facecolor((0,0,0,0)) # Set boxplot to transparent
    #                     patch.set_edgecolor(colors[pos[idx]-1]) # Set boxplot edgecolor to black
                        patch.set_edgecolor((0,0,0,1)) # Set boxplot edgecolor to black

                    for patch in bplot['medians']: 
                        patch.set_color('gold') # Set boxplot median to dark yellow
                        patch.set_linewidth(2) # Set boxplot median line width to 2
            
            except: continue

        plt.xticks(pos, groupsNames)
        # plt.xlim(pos[0]-1, pos[-1]+1)
        plt.ylabel(between)  