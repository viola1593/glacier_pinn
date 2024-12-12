import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
L = 10000  # Glacier length in meters
num_points = 10000  # Number of points along the glacier length
x = np.linspace(0, L, num_points)  # Horizontal grid along glacier length

# Glacier and physical properties
rho = 917          # Density of ice (kg/m^3)
g = 9.81           # Gravitational acceleration (m/s^2)
A = 2.4*1e-24          # Flow law parameter (Pa^-3 s^-1) https://people.maths.ox.ac.uk/hewitt/slides/hewitt_karthaus_rheology.pdf
n = 3              # Glen's flow law exponent
u_b = 0         # Basal velocity (m/s)


# Generate random bed topography and ice thickness
np.random.seed(42)
# Generate bed topography
# Base shape: a valley profile with a downward slope toward the glacier terminus
bed_topography =  -((x-L)/L)**3*300


# Generate realistic ice thickness profile
# Assume thickness is higher near the glacier head and decreases toward the terminus
ice_thickness = 500 * np.exp(-(x / ( L))**2)  # Exponential decay of thickness along glacier length
ice_thickness +=150 * np.sin( np.pi * x / L+0.3) 
ice_thickness += -100 * (1- (x / L )) 	
#surface_elevation = ice_thickness




surface_elevation=ice_thickness +bed_topography

# Surface slope (approximated by finite differences)
surface_slope = np.abs(np.gradient(surface_elevation, x))
basal_stress = rho*g*ice_thickness*surface_slope

# Initialize depth-averaged velocity profile
# Initial velocity based on driving stress and basal drag
velocity = u_b+(2*A)/(n+1)*basal_stress**n*ice_thickness

# Calculate beta values based on slope-velocity ratio
slope_vel_ratio = surface_slope/velocity
beta = [np.tanh(-4000*x+1.5)*0.45+0.55 for x in slope_vel_ratio] 

# calculate apparent mass balance from mass conservation mb = div(vH) 
mb= np.gradient(velocity, x)*ice_thickness+velocity*np.gradient(ice_thickness, x)
print(np.mean(mb*86400*365.24))

# make dataframe to save the values

df = pd.DataFrame()
df['x'] = x
#f['bed_topography'] = bed_topography
df['ice_thickness'] = ice_thickness
df['surface_elevation'] = surface_elevation
df['surface_slope'] = surface_slope
df['surface_velocity'] = velocity*86400*365.24 # in m/a
df['beta'] = beta
df['apparent_mass_balance'] = mb *86400*365.24 # in m²/a




# Plot the generated glacier profiles and velocity
plt.figure(figsize=(12, 16))
plt.subplot(5, 1, 1)
plt.plot(x, bed_topography, label="Bed Topography")
plt.plot(x, surface_elevation, label="Surface Elevation")
plt.fill_between(x, bed_topography, surface_elevation, color='lightblue', alpha=0.5)
plt.ylabel("Elevation (m)")
plt.title("Synthetic Glacier Geometry")
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(x, ice_thickness, color="purple")
plt.ylabel("Ice Thickness (m)")
plt.title("Ice Thickness Profile")

plt.subplot(5, 1, 3)
plt.plot(x, basal_stress*1e-3, color="red")
plt.ylabel("Basal Stress (kPa)")
plt.title("Basal Stress Profile")


plt.subplot(5, 1, 4)
plt.plot(x, velocity*86400*365.24, color="red")
plt.ylabel("Surface Velocity (m/a)")
plt.xlabel("Distance along Glacier (m)")
plt.title("Surface Velocity Profile")

# plt.subplot(6, 1, 5)
# plt.plot(x, beta, color="green")
# plt.ylabel("Beta")
# plt.xlabel("Distance along Glacier (m)")
# plt.title("Beta Profile")

plt.subplot(5, 1, 5)
plt.plot(x, mb*86400*365.24, color="blue")
plt.ylabel("Apparent Mass Balance (m/a)")
plt.xlabel("Distance along Glacier (m)")
plt.title("Apparent Mass Balance Profile")

plt.tight_layout()
#plt.show()
plt.savefig('data/synthetic_data/multipleglacier/1dsynthetic_profile.png')

df.to_csv('data/synthetic_data/1dsynthetic_glacier_profile.csv', index=False)