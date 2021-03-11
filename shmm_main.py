import warnings
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import concurrent.futures
import multiprocessing
import argparse

import stock_info

# Calculating Mean Absolute Percentage Error of predictions
def calc_mape(predicted_data, true_data):
    return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0])


def predict_prices(dataset, num_test=100, K = 50, num_iters=10000):
    print("Defining the model...")
    model, opt_states = define_model(dataset=dataset, num_test=num_test, num_iters=num_iters)
    print("Finished defining the model")
    print("Training the model...")
    predicted_stock_data = train_model(dataset=dataset, model=model, opt_states=opt_states, num_test=num_test, num_iters=num_iters, K=K)
    print("Finished training the model")
    return predicted_stock_data

def define_model(dataset, num_test, num_iters):
    # Possible number of states in Markov Model
    STATE_SPACE = range(2,15)
    likelihood_vect = np.empty([0,1])
    aic_vect = np.empty([0,1])
    bic_vect = np.empty([0,1])

    for states in STATE_SPACE:
        num_params = states**2 + states
        dirichlet_params_states = np.random.randint(1,50,states)
        model = hmm.GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=num_iters)
        model.fit(dataset[num_test:,:])
        if model.monitor_.iter == num_iters:
            print('Increase number of iterations')
            sys.exit(1)
        likelihood_vect = np.vstack((likelihood_vect, model.score(dataset)))
        aic_vect = np.vstack((aic_vect, -2 * model.score(dataset) + 2 * num_params))
        bic_vect = np.vstack((bic_vect, -2 * model.score(dataset) +  num_params * np.log(dataset.shape[0])))
    
    opt_states = np.argmin(bic_vect) + 2
    return (model, opt_states)

def train_model(dataset, model, opt_states, num_test, K, num_iters):
    predicted_stock_data = np.empty([0,dataset.shape[1]])
    train_dataset = dataset[100:,:]
    for idx in reversed(range(num_test)):
            num_examples = train_dataset.shape[0]
            
            if idx == num_test - 1:
                model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=num_iters, init_params='stmc')
            else:
                # Retune the model by using the HMM paramters from the previous iterations as the prior
                model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=num_iters, init_params='')
                model.transmat_ = transmat_retune_prior 
                model.startprob_ = startprob_retune_prior
                model.means_ = means_retune_prior
                model.covars_ = covars_retune_prior

            model.fit(np.flipud(train_dataset))

            transmat_retune_prior = model.transmat_
            startprob_retune_prior = model.startprob_
            means_retune_prior = model.means_
            covars_retune_prior = model.covars_

            if model.monitor_.iter == num_iters:
                print('Increase number of iterations')
                exit(1)

            iters = 1;
            past_likelihood = []
            curr_likelihood = model.score(np.flipud(train_dataset[0:K - 1, :]))
            while iters < num_examples / K - 1:
                past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset[iters:iters + K - 1, :])))
                iters = iters + 1
            likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
            predicted_change = train_dataset[likelihood_diff_idx,:] - train_dataset[likelihood_diff_idx + 1,:]
            new_data = train_dataset[0,:] + predicted_change
            train_dataset = np.insert(train_dataset, 0, new_data, axis=0)

            predicted_stock_data = np.vstack((predicted_stock_data, new_data))
    
    return predicted_stock_data

def plot_data(predicted_stock_data, dataset, ticker, plot_all_features=False):
    labels = ['Close','Open','High','Low']
    if plot_all_features:
        for i in range(4):
            plt.figure()
            plt.plot(range(100), predicted_stock_data[:,i],'k-', label = 'Predicted '+labels[i]+' price');
            plt.plot(range(100),np.flipud(dataset[range(100),i]),'r--', label = 'Actual '+labels[i]+' price')
            plt.xlabel('Time steps')
            plt.ylabel('Price')
            plt.title(labels[i]+' price'+ ' for '+ticker[:-4])
            plt.grid(True)
            plt.legend(loc = 'upper left')
    else:
        hdl_p = plt.plot(np.array(range(100)), np.array(predicted_stock_data));
        plt.title('Predicted stock prices')
        plt.legend(iter(hdl_p), ('Close','Open','High','Low'))
        plt.xlabel('Time steps')
        plt.ylabel('Price')
        plt.figure()
        hdl_a = plt.plot(range(100),np.flipud(dataset[range(100),:]))
        plt.title('Actual stock prices')
        plt.legend(iter(hdl_p), ('Close','Open','High','Low'))
        plt.xlabel('Time steps')
        plt.ylabel('Price')
        

    plt.show()

def get_arg_parser():
    args = argparse.ArgumentParser(description='Analyzes a given stock and its price history to generate predictions for future performance')

    args.add_argument('-t', '--ticker', type=str, help="the stock ticker to analyze")
    args.add_argument('-i', '--sample_internal', nargs='?', const=1440, type=int, help="the sample interval of market data, in minutes. Default is a day")
    args.add_argument('-ni', '--num_input', nargs='?', const=None, type=int, help="the number of data points to be used in the model. Must be >200.")
    args.add_argument('-nr', '--num_results', nargs='?', const=100, type=int, help="the number of data points in the prediction data")
    args.add_argument('-s',
                       '--save',
                       action='store_true',
                       help='save the predicted prices to csv')
    args.add_argument('-np',
                       '--no_plot',
                       action='store_true',
                       help='do not plot the predicted prices')
    args.add_argument('-pa',
                       '--plot_all',
                       action='store_true',
                       help='plot each of the price features individually')
    return args

def main():
    args = get_arg_parser().parse_args()
    ticker = args.ticker
    print("Retrieving data for", ticker)
    dataset = np.genfromtxt('apple.csv', delimiter=',')
    # try:
    #     dataset = stock_info.get_stock_info(ticker=ticker).values
    # except ValueError as e:
    #     print ("Something went wrong while retrieving the price data.")
    #     print ("Error was:", str(e))
    #     print("Exiting.")
    #     exit()
    print("Successfully retrieved data for", ticker)

    subset_size = args.num_input
    if (not subset_size):
        subset_size = len(dataset) - 1
    elif (args.num_input) < 201:
        print ("To sucessfully train the model, there need to be more than 200 data points.\nExiting")
        exit()
    elif (args.num_input > len(dataset)):
        print (f'The number of input: {args.num_input} is greater than the data set size: {len(dataset)}.\nExiting')
        exit()

    data_subset = dataset[:subset_size]
    predicted_data = predict_prices(data_subset)

    if (args.save):
        filename = '{}_forecast.csv'.format(ticker)
        print("Saving to", filename)
        np.savetxt(filename, predicted_data, delimiter=',', fmt='%.2f')

    mape = calc_mape(predicted_data, np.flipud(dataset[range(100),:]))
    print('\nMAPE for the stock {} is '.format(ticker), mape)
        
    if (not args.no_plot):
        plot_data(predicted_stock_data=predicted_data, dataset=dataset, ticker=ticker, plot_all_features=args.plot_all)

if __name__ == "__main__":
    main()
