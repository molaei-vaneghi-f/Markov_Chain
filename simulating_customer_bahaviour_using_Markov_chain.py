import os
import numpy as np
import pandas as pd


#%% read the data

db = './data'
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
data = []
mm = []

for day in days:
    
    df = pd.read_csv(os.path.join(db, day + '.csv'), sep=';', parse_dates=True) 
    df_mm = pd.read_csv(os.path.join(db, 'mm_' + day + '.csv'), index_col = 0) 
    mm.append(df_mm)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    data.append(df)


#%% mc simulation
# Use your transition probability matrix to propagate the states of an customer_idealized population. 
# Assume that there are infinite customers, so you can conscustomer_ider a state distribution.


class Customer:
    
    def __init__(self, idx, state, transition_matrix):
        self.id = idx
        self.state = state
        self.transition_matrix = transition_matrix

        self.transitin_array_dict = {
        'dairy' : self.transition_matrix[0,:],
        'drinks' : self.transition_matrix[1,:],
        'entry' : self.transition_matrix[2,:],
        'fruit' : self.transition_matrix[3,:],
        'spices' : self.transition_matrix[4,:]
        }

    def __repr__(self):
        
        """
        Returns a csv string for a specific customer.
        """
        return f'{self.id};{self.state}'

    def is_active(self):
        
        """
        Returns False if the customer is no longer active (exited the store)
        i.e. has reached the checkout for the second time
        """
        
        if self.state != 'checkout':
             return True 
        if self.state == 'checkout':
            return False 
        
        
    def next_state(self):
        
        """
        using a weighted random choice from the transition probabilities
        propagates the customer to the next state
        conditional on the current state.
        """
  
        self.state = np.random.choice(['checkout', 'dairy', 'drinks', 'fruit', 'spices'], p=self.transitin_array_dict[f'{self.state}'])
    

#%% transition probabilities for one customer
# We would like to analyze how customers switch between sections of the supermarket. 

def transition_probability(df):
    
    """
    Calculates he probability of transitions from section A to B by counting all observed transitions
    given the customer info (df), calculate the prob matrices per customer
    """

    customer_no = 1
    
    for i in range(df["customer_no"].max()):
        
        df_temp = df[df["customer_no"].isin([customer_no])].copy()
        df_temp['location_before'] = df_temp['location'].shift(1)
        df_temp['timestamp_before'] = df_temp['timestamp'].shift(1)
        df_temp['timestamp_diff'] = df_temp['timestamp'] - df_temp['timestamp_before']
        print('Customer number: ' + str(customer_no) + '\n')
        print(pd.crosstab(df_temp['location'], df_temp['location_before'], normalize=1))
        print("*************************** "+ '\n')
        
        customer_no += 1


def make_frequency_df(df_copy):
    
    '''
    Construct a dataframe such that indices are seperated by delta 1 min from the Market Data
    to be used with pd.crosstab() method to obtain markov matrices 
    '''
    
    df = df_copy.copy()
    frames = pd.DataFrame()
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    for customer in df['customer_no'].unique():

        df_tmp = df[df['customer_no'] == customer]
        
        #expand timestamp index such that delta T is 1 min, and forward fill isles
        df_tmp = df_tmp.asfreq('T',method='ffill')
        
        #insert 'entry' 1 min before first isle 
        df_tmp.location[df_tmp.index[0] - pd.to_timedelta('1min')] = [customer,'entry']
        df_tmp.sort_index(inplace=True)
        
        #after is simply a shift(-1) of current location
        df_tmp['after'] = df_tmp['location'].shift(-1)
        df_tmp.dropna(inplace=True)
        
        frames = pd.concat([frames, df_tmp], axis=0)
        
    return frames

def create_markov_matrix(frames_df):
    '''
    Generate the Markov Matrix for a Market Data dataframe, structured by make_frequency_df function
    NOTE: Columns indicate current state, rows indicate after state, probabilities are read current -> after probability
    sum of columns should add to 1
    '''
    df = frames_df.copy()
    
    return pd.crosstab(df['after'], df['location'], normalize=1)



trans_prob = transition_probability(data[1])
frequency_table_df = make_frequency_df(df)
m_matrix = create_markov_matrix(frequency_table_df)

c1 = Customer(1, 'fruit', np.array(m_matrix))
# c1 = Customer(1, 'fruit', np.array(mm[0]))
c1.next_state()
c1.state

