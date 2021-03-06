import warnings
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import concurrent.futures

import stock_info


likelihood_vect = np.empty([0,1])
aic_vect = np.empty([0,1])
bic_vect = np.empty([0,1])

# Possible number of states in Markov Model
STATE_SPACE = range(2,15)

# Calculating Mean Absolute Percentage Error of predictions
def calc_mape(predicted_data, true_data):
    return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0])


def predict_prices(dataset, NUM_TEST=100, K = 50, NUM_ITERS=10000):
    model, opt_states = define_model(dataset=dataset, NUM_TEST=NUM_TEST, NUM_ITERS=NUM_ITERS)

    predicted_stock_data = train_model(dataset=dataset, model=model, opt_states=opt_states, NUM_TEST=NUM_TEST, NUM_ITERS=NUM_ITERS, K=K)

    return predicted_stock_data

def define_model(dataset, NUM_TEST, NUM_ITERS):
    likelihood_vect = np.empty([0,1])
    aic_vect = np.empty([0,1])
    bic_vect = np.empty([0,1])
    for states in STATE_SPACE:
        num_params = states**2 + states
        dirichlet_params_states = np.random.randint(1,50,states)
        model = hmm.GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS)
        model.fit(dataset[NUM_TEST:,:])
        if model.monitor_.iter == NUM_ITERS:
            print('Increase number of iterations')
            sys.exit(1)
        likelihood_vect = np.vstack((likelihood_vect, model.score(dataset)))
        aic_vect = np.vstack((aic_vect, -2 * model.score(dataset) + 2 * num_params))
        bic_vect = np.vstack((bic_vect, -2 * model.score(dataset) +  num_params * np.log(dataset.shape[0])))
    
    opt_states = np.argmin(bic_vect) + 2
    print('\nOptimum number of states are {}'.format(opt_states))
    return (model, opt_states)

def train_model(dataset, model, opt_states, NUM_TEST, K, NUM_ITERS):
    predicted_stock_data = np.empty([0,dataset.shape[1]])

    for idx in reversed(range(NUM_TEST)):
            train_dataset = dataset[idx + 1:,:]
            test_data = dataset[idx,:]; 
            num_examples = train_dataset.shape[0]
            
            if idx == NUM_TEST - 1:
                model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS, init_params='stmc')
            else:
                # Retune the model by using the HMM paramters from the previous iterations as the prior
                model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS, init_params='')
                model.transmat_ = transmat_retune_prior 
                model.startprob_ = startprob_retune_prior
                model.means_ = means_retune_prior
                model.covars_ = covars_retune_prior

            model.fit(np.flipud(train_dataset))

            transmat_retune_prior = model.transmat_
            startprob_retune_prior = model.startprob_
            means_retune_prior = model.means_
            covars_retune_prior = model.covars_

            if model.monitor_.iter == NUM_ITERS:
                print('Increase number of iterations')
                sys.exit(1)
            print('Model score : ', model.score(dataset))

            iters = 1;
            past_likelihood = []
            curr_likelihood = model.score(np.flipud(train_dataset[0:K - 1, :]))
            while iters < num_examples / K - 1:
                past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_dataset[iters:iters + K - 1, :])))
                iters = iters + 1
            likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
            predicted_change = train_dataset[likelihood_diff_idx,:] - train_dataset[likelihood_diff_idx + 1,:]
            predicted_stock_data = np.vstack((predicted_stock_data, dataset[idx + 1,:] + predicted_change))

    return predicted_stock_data

def plot_data(predicted_stock_data, dataset, symbol, FOR_EACH_LABEL=False):
    labels = ['Close','Open','High','Low']
    if FOR_EACH_LABEL:
        for i in range(4):
            plt.figure()
            plt.plot(range(100), predicted_stock_data[:,i],'k-', label = 'Predicted '+labels[i]+' price');
            plt.plot(range(100),np.flipud(dataset[range(100),i]),'r--', label = 'Actual '+labels[i]+' price')
            plt.xlabel('Time steps')
            plt.ylabel('Price')
            plt.title(labels[i]+' price'+ ' for '+symbol[:-4])
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

def animated_loading():
    chars = "/—\|" 
    for char in chars:
        sys.stdout.write('\r'+'loading...'+char)
        time.sleep(.1)
        sys.stdout.flush() 

def main():
    symbol = "tsla"
    dataset = stock_info.get_stock_info(symbol=symbol).values

    # dataset = np.genfromtxt('apple.csv', delimiter=',')
    # result =np.genfromtxt('apple.csv_forecast.csv', delimiter=',')
    # data.to_csv('andrew.csv', sep=',', header=False, index=False)
    # plot_data(predicted_stock_data=result, dataset=dataset, symbol=symbol)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(predict_prices, dataset)

        while(future.running()):
            animated_loading()

        np.savetxt('{}_forecast.csv'.format(symbol),future.result(),delimiter=',',fmt='%.2f')

        mape = calc_mape(future.result(), np.flipud(dataset[range(100),:]))
        print('\nMAPE for the stock {} is '.format(symbol),mape)
            
        plot_data(predicted_stock_data=future.result(), dataset=dataset, symbol=symbol)

if __name__ == "__main__":
    main()
