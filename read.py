import pandas as pd
import glob

def readData(type,year):

    path = r'data/'
    all_files = glob.glob(path +year+'*\*_'+type+'.xlsx')

    li = []

    for filename in all_files:
        df = pd.read_excel(filename)
        df = df.rename(columns={'MpKrakowWIOSPrad6115': 'n'})
        if (df.columns[0]=='Unnamed: 0' or df.columns[0]=='Nr'):
            new_header = df.iloc[0]  # grab the first row for the header
            new_header[0] = "Czas"
            df = df[5:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
        else:
            df = df[1:]
            if ("2012_PM10_1g" not in filename): #brakuje jednego wiersza w tym konkretnym pliku
                df = df[1:]
        df = df.rename(columns={'Kod stacji': 'Czas'})
        df = df.rename(columns={'Kod stanowiska': 'Czas'})
        df = df.rename(columns={'MpKrakowWIOSAKra6117': 'MpKrakAlKras'})
        df = df.rename(columns={'MpKrakowWIOSBuja6119': 'MpKrakBujaka'})
        df = df.rename(columns={'MpKrakowWIOSBulw6118': 'MpKrakBulwar'})
        df=df.filter(regex=("MpKrak.*|Czas.*"),axis=1)
        df['Czas'] = pd.to_datetime(df['Czas'])
        for col in df.columns:
            if col != 'Czas':
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float).interpolate(limit_area="inside").astype("Int64",errors='ignore')
        print(filename)
        print(df.info())
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True).set_index('Czas')
