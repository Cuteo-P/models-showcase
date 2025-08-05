import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np

class DataPreprocess:
    def __init__(self, data):
        self.data = data
            
    def FeatureEngineering(self):
        self.data['brand_model'] = self.data['brand'].str.strip() + ' ' + self.data['model'].str.strip()
        self.data = self.data.drop(columns=['model', 'year'])
        self.data['miles_per_year'] = (self.data['mileage'] / self.data['car_age']).round(2)
        return self
    
    def OHE(self):
        oh_encoder = OneHotEncoder(sparse_output=False)
        encoded_data = oh_encoder.fit_transform(self.data[['fuel_type', 'color']])
        df_encoded = pd.DataFrame(encoded_data, columns=oh_encoder.get_feature_names_out(['fuel_type', 'color']))
        df_encoded = df_encoded.reset_index(drop=True)
        self.data = self.data.reset_index(drop=True)
        self.data = pd.concat([self.data, df_encoded], axis=1)
        self.data.drop(columns=['fuel_type', 'color'], inplace=True)
        return self
    
    def OrdinalEncoding(self):
        categorical_cols_ordinal = ['brand', 'owner_type', 'transmission', 'brand_model']
        category_order = [
            ['Lexus', 'Porsche', 'Audi', 'Volvo', 'Volkswagen', 'Nissan', 'BMW', 'Chevrolet', 'Toyota', 'Mercedes',
             'Subaru', 'Jaguar', 'Ford', 'Kia', 'Jeep', 'Mazda', 'Hyundai', 'Honda', 'Land Rover', 'Tesla'],
            ['First', 'Second', 'Third'],
            ['Manual', 'Automatic'],
            ['Lexus GX', 'Porsche Macan', 'Audi A6', 'Volvo XC40', 'Volkswagen Atlas', 'Volvo XC60', 'Volkswagen Jetta',
             'Nissan Leaf', 'Audi Q5', 'BMW 5 Series', 'Chevrolet Malibu', 'Porsche Taycan', 'Toyota Prius',
             'Mercedes C-Class', 'Subaru Forester', 'Subaru Legacy', 'Jaguar F-PACE', 'Subaru Impreza', 'Nissan Rogue',
             'Nissan Maxima', 'Kia Sportage', 'Toyota Corolla', 'Jeep Grand Cherokee', 'Volkswagen Tiguan', 'Toyota RAV4',
             'Mazda Mazda3', 'Chevrolet Camaro', 'Ford Focus', 'Mazda CX-30', 'Kia Forte', 'Porsche Panamera', 'BMW i8',
             'Hyundai Elantra', 'Hyundai Kona', 'Lexus IS', 'Audi Q7', 'Audi A8', 'Jeep Renegade', 'Jaguar XE', 'Lexus NX',
             'Mazda CX-5', 'Hyundai Sonata', 'Mercedes S-Class', 'Subaru Crosstrek', 'Toyota Camry', 'Nissan Altima',
             'Mazda Mazda6', 'Jaguar F-TYPE', 'Honda Pilot', 'Honda HR-V', 'Mercedes GLC', 'Volvo V60', 'Porsche 911',
             'Kia Optima', 'Hyundai Santa Fe', 'Chevrolet Equinox', 'Jaguar I-PACE', 'Jeep Cherokee', 'Lexus RX',
             'Hyundai Tucson', 'Ford Explorer', 'BMW X3', 'Kia Soul', 'Audi A3', 'BMW X5', 'Chevrolet Silverado',
             'Toyota Highlander', 'Honda Accord', 'Kia Sorento', 'Mercedes E-Class', 'Land Rover Range Rover',
             'Honda Civic', 'Jeep Wrangler', 'Ford Mustang', 'Chevrolet Traverse', 'Volkswagen Passat', 'Volvo S60',
             'Nissan Sentra', 'Mazda CX-9', 'Subaru Outback', 'Mercedes GLE', 'Lexus ES', 'Ford F150', 'Volvo XC90',
             'Tesla Model Y', 'BMW 3 Series', 'Land Rover Discovery', 'Tesla Model 3', 'Jaguar XF', 'Jeep Compass',
             'Ford Fusion', 'Volkswagen Golf', 'Honda CR-V', 'Porsche Cayenne', 'Tesla Model X', 'Tesla Model S']
        ]
        encoder = OrdinalEncoder(categories=category_order)
        self.data[categorical_cols_ordinal] = encoder.fit_transform(self.data[categorical_cols_ordinal])
        return self
    
    def TargetModification(self):
        self.data = self.data[self.data['price'] != 0]
        self.data['price'] = np.log1p(self.data['price'])
        return self