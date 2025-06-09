import pvlib
import pandas as pd
import matplotlib.pyplot as plt

#Setting the location = Nottingham
nottingham  = pvlib.location.Location(
    latitude= 52.94, longitude = -1.19, tz = "Europe/London",altitude = 61, name = "Nottingham")

#Getting TMY data for nottingham
tmy_data, _, _, _ =pvlib.iotools.get_pvgis_tmy(
    latitude = nottingham.latitude,
    longitude = nottingham.longitude,
    map_variables = True
)
tmy_data = tmy_data.rename(columns={"Gb(n)": "ghi", "G(h)": "dni", "Gd(h)": "dhi"})

#Creating a PV System
system = pvlib.pvsystem.PVSystem(
    surface_tilt=30,
    surface_azimuth=180,
    module_parameters = {
        'pdc0': 300, #300W module
        'gamma_pdc': -0.004, #temperature coefficient
        'K' : 4.0,  #needed for physical aoi model
        'L' : 0.002, #Needed for physical aoi model
        'a' : -3.47,
        'b' : -0.0594,
        'deltaT': 3.0
    },
    inverter_parameters = {'pdc0': 300},
    racking_model= 'open_rack',
    module_type = 'glass_polymer'
)

#Running model chain
mc = pvlib.modelchain.ModelChain(
    system = system,
    location = nottingham,
    aoi_model = 'physical',
    spectral_model = 'no_loss',
    temperature_model = 'sapm',
)
mc.run_model(tmy_data)

#Create DataFrame for the results with full year's data
results_df = pd.DataFrame({
    'AC Power (W)': mc.results.ac,
    'Temperature (C)': mc.results.cell_temperature,
    'Irradiance (W/m^2)': mc.results.effective_irradiance
})

#Normalised index to a consistent year (2020)
results_df.index = pd.date_range(start='2020-01-01', periods=len(results_df), freq='H')

#Showing results
print("Full Year's Simulated Data:")
print(results_df)

#Save to CSV
results_df.to_csv('nottingham_pv_year.csv', index=False)

#Plotting
plt.figure(figsize=(15, 10))

#Plot AC Power
plt.subplot(3, 1, 1)
results_df['AC Power (W)'].plot(title='AC Power Output - Annual Profile', xlabel='Time', ylabel='Power (W)')

#Plot Temperature
plt.subplot(3, 1, 2)
results_df['Temperature (C)'].plot(title='Cell Temperature - Annual Profile', xlabel='Time', ylabel='Temperature (°C)', color = 'r')

#Plot Irradiance
plt.subplot(3, 1, 3)
results_df['Irradiance (W/m^2)'].plot(title='Effective Irradiance - Annual Profile', xlabel='Time', ylabel='Irradiance (W/m²)', color = 'g')

plt.tight_layout()
plt.show()


#print(pd.DataFrame({
   # 'AC Power (W)': mc.results.ac,
   # 'Temperature (C)': mc.results.cell_temperature,
    #'Irradiance (W/m^2)': mc.results.effective_irradiance,
#}).head(24))


