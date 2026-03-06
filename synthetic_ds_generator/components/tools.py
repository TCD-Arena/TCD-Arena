import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math




def human_readable_labels(labels):
    """
    Human readable labels for the Causal Graph and TS.
    """
    a = pd.DataFrame(labels.sum(axis=2))
    return pd.concat(
        [pd.concat(
            [a],
            keys=['Cause'], axis=1)],
        keys=['Effect']
    )



def predict_moving_window(m,test,exog=None, history=5, forecast=10):
 
    stack = []
    for x in range(len(test) - (forecast+history)):
        bg = test[:x+history]
        predict = test[x+history:x+forecast+history]
        if isinstance(exog, np.ndarray):
            context = exog[x+history:x+forecast+history]
        pred = m.forecast(bg, forecast, exog_future=(
            context if isinstance(exog, np.ndarray) else None))
        error = np.abs(predict - pred).mean(axis=0)
        stack.append(error)
    stack = np.array(stack)
    return stack.mean(axis=0)


def visualize_nl_function(ax,components,coefficient, range=[-10,10], granularity=0.1): 
    """
    takes in a stack of nl function selections
      and a dict that specifies the functions and display
        the relationship in a specified window
    """

    Y =  []
    X = np.arange(range[0], range[1],granularity)
    for x in X:
        Y.append(np.mean([c(x) for c in components]))
    Y = np.array(Y)
    Y = Y*coefficient
    ax.plot(X,Y)


def visualize_functional_relationships(links, nl, lag=1, nl_opt= None):
    """
    Visualizes the functional relationships of a process for a certain lag. 

    """

    fig, axs = plt.subplots(figsize = (10,10), *links.shape[:-1])

    relevant_l = links[:,:,lag-1]
    relevant_nl = np.stack(nl)[:,:,:,lag-1]
    eff,cau = relevant_l.shape

    for x in range(eff): 
        for y in range(cau):
            if relevant_l[x,y] == 0:
                pass
            else:
                funcs = []
                for item in  relevant_nl[:,x,y]:
                    if item == -1: 
                        funcs.append(lambda a : a) # linear component
                    else:
                        funcs.append(nl_opt[item])

                visualize_nl_function(axs[x,y], funcs, coefficient=relevant_l[x,y])
    fig.suptitle("Relationship for lag = {}".format(lag))
    print(relevant_l.shape)


def visualize_exogs(links, nl, nl_opt= None):
    """
    Visualizes the functional relationships of a process for a certain lag. 

    """

    fig, axs = plt.subplots(figsize = (10,10), *links.shape)
    relevant_nl = np.stack(nl)
    eff,cau = links.shape

    for x in range(eff): 
        for y in range(cau):
            if links[x,y] == 0:
                pass
            else:
                funcs = []
                for item in  relevant_nl[:,x,y]:
                    if item == -1: 
                        funcs.append(lambda a : a) # linear component
                    else:
                        funcs.append(nl_opt[item])

                visualize_nl_function(axs[x,y], funcs ,range=[-3,3],coefficient=links[x,y])
    fig.suptitle("Exogenous influences")
    print(links.shape)


def calc_steps_according_to_change_points(change_at, time_series_n):
    # Calculate the number of steps to take.
    if len(change_at) == 0: 
        return [time_series_n]
    else:
        change_points = [0] + change_at + [time_series_n]
        steps = [change_points[x] - change_points[x-1] for x in range(1, len(change_points))]
        return steps

    
def step_increase(x, l=10,r=20):
    # 30 interval for months. needs more work.
    
    if x%30 >= l and x%30 <= r:
        return 3
    else:
        return 1
    
def mean_var_shift(step, scale=0.001):
    fraction_of_year = step / 730 # need 730 to not get 2 cycles per year.
    angle_in_radians = 2 * math.pi * fraction_of_year
    
    # Calculate the sine of the angle
    sine_value = math.sin(angle_in_radians)
    return sine_value* (1 + (step * scale))
    
    
def scale_up_through_time(step, scale=0.01):
    """
    Monotonically incrases the value of the step
    """    
    return 1 + (step * scale) 
   


def sinus_year_cycle(day_of_year, scale=1):
    """
    Calculates the sine value for a given day of the year, 
    creating a sinusoidal cycle over a 365-day period.
    """
    fraction_of_year = day_of_year / 730 # need 730 to not get 2 cycles per year.
    angle_in_radians = 2 * math.pi * fraction_of_year
    
    # Calculate the sine of the angle
    sine_value = math.sin(angle_in_radians)
    
    return sine_value* scale

def check_divergence(data, th=10, maximum=25):
    # Check, if for the past 10 timesteps, one variable only increased
    if data.shape[1]<th:
        return False
    data = np.abs(data[:,-th:])
    increase = data[:, :-1] - data[:,1:]
    # if for th steps, the graph is monotonically increasing 
    # for any variable we consider it to be divergent
    res = np.all(increase < 0, axis=1)
    return np.any(res) or (np.abs(data).max() > maximum)



def normalize_ts(ts):
    ts = (ts  - np.expand_dims(
        ts.min(axis=1),1)) /  (np.expand_dims(ts.max(
            axis=1),1) - np.expand_dims(ts.min(axis=1),1))
    
    return ts


def check_weibull_stats():
    import numpy as np
    from scipy.special import gamma # The Gamma function
    # --- 1. Define Parameters and Generate Data ---
    shape_k = 1.5
    size = 1000000 # Use a large size for better accuracy

    # Generate samples for a Weibull(shape=k, scale=lambda) distribution
    weibull_samples = np.random.weibull(shape_k, size)


    # --- 2. Calculate Sample Mean and Variance from the Data ---
    sample_mean = np.mean(weibull_samples)
    sample_var = np.var(weibull_samples)

    print("--- From Generated Data (Sample) ---")
    print(f"Sample Mean:      {sample_mean:.4f}")
    print(f"Sample Variance:  {sample_var:.4f}")
    print("-" * 35)


    # --- 3. Calculate Theoretical Mean and Variance from Formulas ---
    theoretical_mean = gamma(1 + 1/shape_k)
    theoretical_var =  (gamma(1 + 2/shape_k) - (gamma(1 + 1/shape_k))**2)

    print("--- From Formulas (Theoretical) ---")
    print(f"Theoretical Mean:     {theoretical_mean:.4f}")
    print(f"Theoretical Variance: {theoretical_var:.4f}")
    print("-" * 35)

    