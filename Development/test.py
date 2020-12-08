
#numar valori diferite / coloana
for c in df.columns:
    print("---- %s ---" % c)
    print(df[c].value_counts())
    print()

#numar entry-uri null
print(df['rezultat testare'].isnull().sum())
