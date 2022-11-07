import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object ):
    def __init__( self ):
        self.home_path = 'C:/Users/igor/Documents/repos/ds_em_producao/Project/'
        self.competition_distance_scaler =  pickle.load(open( self.home_path  +  'parameter/competition_distance_scaler.pkl',          'rb'))
        self.competition_time_month_scaler =       pickle.load(open( self.home_path  +  'parameter/competition_time_month_scaler.pkl',        'rb'))
        self.promo_time_week_scaler =              pickle.load(open( self.home_path  +  'parameter/promo_time_week_scaler.pkl',               'rb'))
        self.year_scaler =                  pickle.load(open( self.home_path  +  'parameter/year_scaler.pkl',                          'rb'))
        self.store_type_scaler =            pickle.load(open( self.home_path  +  'parameter/store_type_scaler.pkl',                    'rb'))
    
    def data_cleaning(self, df1):
        # Rename columns
        old_cols = ['Store', 'DayOfWeek', 'Date','Open','Promo', 'StateHoliday', 'SchoolHoliday',
                              'StoreType', 'Assortment', 'CompetitionDistance','CompetitionOpenSinceMonth',
                              'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek','Promo2SinceYear', 'PromoInterval']
                              
        snakecase = lambda x: inflection.underscore( x )
        new_cols = list ( map( snakecase, old_cols))
        # rename
        df1.columns = new_cols

        # 1.3. Data Types
        df1['date'] = pd.to_datetime(df1['date'])
        # 1.5. Fillout NA
        # competition_distance
        df1['competition_distance'] = df1['competition_distance'].fillna(200000)
        # competition_open_since_month
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if np.isnan(x['competition_open_since_month']) == True else x['competition_open_since_month'], axis = 1)
        # competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if np.isnan(x['competition_open_since_year']) == True else x['competition_open_since_year'], axis = 1)
        # promo2_since_week
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if np.isnan(x['promo2_since_week']) == True else x['promo2_since_week'], axis = 1)
        # promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if np.isnan(x['promo2_since_year']) == True else x['promo2_since_year'], axis = 1)
        # promo_interval
        months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df1['promo_interval'].fillna(0, inplace = True)
        df1['month_map'] = df1['date'].dt.month.apply(lambda x: months[x])
        df1['is_promo'] = df1.apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis = 1)
        ### 1.6. Change Types
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] =  df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1

    def feature_engineering(self, df2):
        # year
        df2['year'] = df2['date'].dt.year
        # month
        df2['month'] = df2['date'].dt.month
        # day
        df2['day'] = df2['date'].dt.day
        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear
        # year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year = x['competition_open_since_year'], month = x['competition_open_since_month'], day = 1), axis = 1)
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since']) / 30).apply(lambda x: x.days).astype(int)
        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days = 7))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since']) / 7).apply(lambda x: x.days).astype(int)
        # assortment
        dict_assort = {'a': 'basic', 'b': 'extra','c': 'extended'}
        df2['assortment'] = df2['assortment'].apply(lambda x: dict_assort[x])
        # state holiday
        dict_holiday = {'0': 'regular_day', 'a':'public_holiday','b': 'easter','c': 'christmas'}
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: dict_holiday[x])
        ### 3.1. Filtragem das linhas
        df2 = df2[df2['open'] != 0 ]
        ### 3.2 Seleção das colunas
        cols_drop = ["open","promo_interval","month_map"]
        df2 = df2.drop(cols_drop, axis = 1)

        return df2

    def data_preparation(self, df6):
        numerical = df6.select_dtypes(include = ['int32', 'int64', 'float64'])
        categorical = df6.select_dtypes(exclude = ['int32', 'int64', 'float64'])
        ### 5.2. Rescaling
        #competition_distance
        df6['competition_distance'] = self.competition_distance_scaler.transform(df6[['competition_distance']].values)
        # Competition_time_mont
        df6['competition_time_month'] = self.competition_time_month_scaler.transform(df6[['competition_time_month']].values)
        # Promo_time_week
        df6['promo_time_week'] = self.promo_time_week_scaler.transform(df6[['promo_time_week']].values)
        # Year
        df6['year'] = self.year_scaler.transform(df6[['year']].values)
        #### 5.3.1. Encoding
        #state_holiday (one hot encoding)
        df6 = pd.get_dummies(df6, prefix = ['state_holiday'], columns = ['state_holiday'])
        #stote type (label encoding)
        df6['store_type'] = self.store_type_scaler.transform(df6['store_type'])
        #assortment
        assortment_dict = {'basic':1,'extra':2,'extended':3}
        #### 5.3.1. Encoding
        df6['assortment'] = df6['assortment'].map(assortment_dict)
        #### 5.3.3. Nature transformation
        # day_of_week
        df6['day_of_week_sin'] = df6['month'].apply(lambda x: np.sin( x * ( 2. * np.pi / 7 )))
        df6['day_of_week_cos'] = df6['month'].apply(lambda x: np.sin( x * ( 2. * np.pi / 7 )))
        # month
        df6['month_sin'] = df6['month'].apply(lambda x: np.sin( x * ( 2. * np.pi / 12 )))
        df6['month_cos'] = df6['month'].apply(lambda x: np.sin( x * ( 2. * np.pi / 12 )))
        # day 
        df6['day_sin'] = df6['day'].apply(lambda x: np.sin( x * ( 2. * np.pi / 30 )))
        df6['day_cos'] = df6['day'].apply(lambda x: np.sin( x * ( 2. * np.pi / 30 )))
        # week of year
        df6['week_of_year_sin'] = df6['week_of_year'].apply(lambda x: np.sin( x * ( 2. * np.pi / 52 )))
        df6['week_of_year_cos'] = df6['week_of_year'].apply(lambda x: np.sin( x * ( 2. * np.pi / 52 )))

        final_features = ['day_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_cos', 'month_sin', 'week_of_year_cos', 'week_of_year_sin', 
                  'store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 
                  'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'day_sin']

        return df6[final_features]
    
    def get_prediction( self, model, original_data, test_data):
        pred = model.predict(test_data)
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient = 'records', date_format = 'iso')
