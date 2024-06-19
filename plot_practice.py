import matplotlib.pyplot as plt
def plot_lyapunov_exponents(frequencies_centralised,exponents_centralised,frequencies_distributed,exponents_distributed):
    plt.figure(figsize=(10, 10))
    plt.plot(frequencies_centralised,exponents_centralised, 'bo',label="centralised")
    plt.legend()
    plt.xlabel('Frequencies')
    plt.ylabel('Largest Lyapunov exponent')
    plt.title("Largest Lyapunov exponent vs Frequencies")
    plt.plot(frequencies_distributed,exponents_distributed, 'ro',label="distributed")
    plt.legend(loc="upper left")
    plt.axhline(0, color='grey')
    plt.show()
def main():
    frequencies_centralised = [140,220,350,370,400]
    exponents_centralised = [31.366043707557747, 25.761331293836875, -169.26927453773035, 56.75689664735796, -2.6033890825026225]
    frequencies_distributed = [140, 220, 350, 370, 400, 500, 600, 800, 1000, 1120, 1200, 1440, 1800, 2000]
    exponents_distributed = [-68.55798371750004, -55.82777611419666, -30.605820436206177, 61.66218934938799, -48.88544520178574, 29.64488570360812, -14.426049669889947, -28.373736383683138, 44.5788723490416, 22.206805953374868, 79.64221488384146, 48.764868178785456, 38.63527819737343, -5.957716030461899]
    plot_lyapunov_exponents(frequencies_centralised,exponents_centralised,frequencies_distributed,exponents_distributed)
if __name__ == "__main__":
    main()