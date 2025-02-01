# -*- coding: utf-8 -*-
"""
This module is for modelling an inertial actuator
"""
import numpy as np
# k = 42992  # spring constant
# m = 1.21  # moving mass
# zeta = 0.05  # damping ratio


# BL = 5.6  # voice coil coefficient
# R = 3.5 + 1.7  # coil + internal resistance
# L = 0.298e-3  # coil inductance


class Shaker:

    def __init__(self, k=42992, m=1.21, zeta=0.05, Bl=5.6, R=5.2, L=0.298e-3, V_ext=1):
        """
        Create inertial shaker model. The default model is Dataphysics IV40

        Parameters
        ----------
        k : float, optional
            Spring constant. The default is 42992.
        m : float, optional
            Inertial mass. The default is 1.21.
        zeta : float, optional
            Damping ratio. The default is 0.05.
        Bl : float, optional
            Voice coil coefficient. The default is 5.6.
        R : float, optional
            Coil resistance. The default is 5.2.
        L : float, optional
            Coil inductance. The default is 0.298e-3.
        V_ext : float, optional
            Input voltage. The default is 1.

        Returns
        -------
        None.

        """

        c_c = 2*np.sqrt(k*m)
        self.c_viscous = c_c*zeta  # damping ratio

        self.k = k
        self.m = m
        self.Bl = Bl
        self.R = R
        self.L = L

        self.V_ext = V_ext

    def newtons_per_amp(self, f):
        """
        Returns the base blocked current-to-force

        Parameters
        ----------
        f : float or array1D
            frequency range.

        Returns
        -------
        float or array1D
            force array for 1A current.

        """
        omega = 2*np.pi*f

        return -1j*omega*self.m*(self.Bl)/(1j*omega*self.m + self.c_viscous + self.k/(1j*omega))

    # def volts_per_amp(self, f):
    #     """
    #     Returns the base blocked volts per amp

    #     Parameters
    #     ----------
    #     f : float or array1D
    #         frequency range.

    #     Returns
    #     -------
    #     float or array1D
    #         volts array for 1A current.

    #     """
    #     omega = 2*np.pi*f
    #     return (self.R + 1j*omega*self.L) + ((self.Bl)**2)/(1j*omega*self.m + self.c_viscous + (self.k/(1j*omega)))

    def volts_per_amp(self, f):
        """
        Returns the base blocked volts per amp

        Parameters
        ----------
        f : float or array1D
            frequency range.

        Returns
        -------
        float or array1D
            volts array for 1A current.

        """
        omega = 2*np.pi*f
        return (self.R + 1j*omega*self.L) + (1/(1j*omega))*(self.Bl)**2 * self.accelerance(f)

    def newtons_per_volt(self, f):
        """
        Returns the base blocked force per volt

        Parameters
        ----------
        f : float or array1D
            frequency range.

        Returns
        -------
        float or array1D
            force array for 1V.

        """

        return self.newtons_per_amp(f)/self.volts_per_amp(f)

    def accelerance(self, f):
        """


        Parameters
        ----------
        f : float or array1D
            frequency range.

        Returns
        -------
        a : float or array1D
            accelerance.

        """

        omega = 2*np.pi*f

        a = (-omega**2)/((-omega**2)*self.m + 1j*omega*self.c_viscous + self.k)

        return a

    def force(self, f, q0):
        """
        The force applied to the pipe for the given 

        Parameters
        ----------
        f : float or array1D
            frequency.
        q0 : float or array1D
            input displacement where the actuator is applied.

        Returns
        -------
        force_pipe : float or array1D
            the force applied to the pipe at each frequency.

        """

        accelerance_wfe = (2*np.pi*1j*f)**2 * q0
        accelerance_shaker = self.accelerance(f)

        force_per_volt = self.newtons_per_volt(f)

        fwfe_per_fshaker = accelerance_shaker / \
            (accelerance_shaker + accelerance_wfe)

        force_pipe = fwfe_per_fshaker*force_per_volt*self.V_ext

        return force_pipe


#####################################################################

class Shaker2:

    def __init__(self, k=42992, m=1.21, zeta=0.05, Bl=5.6, R=5.2, L=0.298e-3, V_ext=1):
        """
        Create inertial shaker model. The default model is Dataphysics IV40

        Parameters
        ----------
        k : float, optional
            Spring constant. The default is 42992.
        m : float, optional
            Inertial mass. The default is 1.21.
        zeta : float, optional
            Damping ratio. The default is 0.05.
        Bl : float, optional
            Voice coil coefficient. The default is 5.6.
        R : float, optional
            Coil resistance. The default is 5.2.
        L : float, optional
            Coil inductance. The default is 0.298e-3.
        V_ext : float, optional
            Input voltage. The default is 1.

        Returns
        -------
        None.

        """

        c_c = 2*np.sqrt(k*m)
        self.c_viscous = c_c*zeta  # damping ratio

        self.k = k
        self.m = m
        self.Bl = Bl
        self.R = R
        self.L = L

        self.V_ext = V_ext

    def newtons_per_amp(self, f):
        """
        Returns the base blocked current-to-force

        Parameters
        ----------
        f : float or array1D
            frequency range.

        Returns
        -------
        float or array1D
            force array for 1A current.

        """
        omega = 2*np.pi*f

        return -1j*omega*self.m*(self.Bl)/(1j*omega*self.m + self.c_viscous + self.k/(1j*omega))

    # def volts_per_amp(self, f):
    #     """
    #     Returns the base blocked volts per amp

    #     Parameters
    #     ----------
    #     f : float or array1D
    #         frequency range.

    #     Returns
    #     -------
    #     float or array1D
    #         volts array for 1A current.

    #     """
    #     omega = 2*np.pi*f
    #     return (self.R + 1j*omega*self.L) + ((self.Bl)**2)/(1j*omega*self.m + self.c_viscous + (self.k/(1j*omega)))

    def volts_per_amp(self, f, q0=None):
        """
        Returns the base blocked volts per amp

        Parameters
        ----------
        f : float or array1D
            frequency range.

        Returns
        -------
        float or array1D
            volts array for 1A current.

        """
        omega = 2*np.pi*f
        aa = self.accelerance(f)
        if q0 is not None:
            a0 = (2*np.pi*1j*f)**2 * q0

            print("coupled volts per amp")

            accel = a0/(1 + a0/aa)

            print(np.mean(abs(accel)), np.mean(abs(a0)), np.mean(abs(aa)))

            print(np.mean(abs((1/(1j*omega))*(self.Bl)**2 * aa)))
            print(np.mean(abs((self.R + 1j*omega*self.L))))
            return (self.R + 1j*omega*self.L) + (1/(1j*omega))*(self.Bl)**2 * accel

        else:
            return (self.R + 1j*omega*self.L) + (1/(1j*omega))*(self.Bl)**2 * aa

    def newtons_per_volt(self, f, q0=None):
        """
        Returns the base blocked force per volt

        Parameters
        ----------
        f : float or array1D
            frequency range.

        Returns
        -------
        float or array1D
            force array for 1V.

        """

        return self.newtons_per_amp(f)/self.volts_per_amp(f, q0=q0)

    def accelerance(self, f):
        """


        Parameters
        ----------
        f : float or array1D
            frequency range.

        Returns
        -------
        a : float or array1D
            accelerance.

        """

        omega = 2*np.pi*f

        a = (-omega**2)/((-omega**2)*self.m + 1j*omega*self.c_viscous + self.k)

        return a

    def force(self, f, q0):
        """
        The force applied to the pipe for the given 

        Parameters
        ----------
        f : float or array1D
            frequency.
        q0 : float or array1D
            input displacement where the actuator is applied.

        Returns
        -------
        force_pipe : float or array1D
            the force applied to the pipe at each frequency.

        """

        accelerance_wfe = (2*np.pi*1j*f)**2 * q0
        accelerance_shaker = self.accelerance(f)

        force_per_volt = self.newtons_per_volt(f, q0=q0)

        fwfe_per_fshaker = accelerance_shaker / \
            (accelerance_shaker + accelerance_wfe)

        force_pipe = fwfe_per_fshaker*force_per_volt*self.V_ext

        return force_pipe

#################################################################################


class timshaker:

    def __init__(self, k=42992, m=1.21, zeta=0.05, Bl=5.6, R=5.2, L=0.298e-3, V_ext=1):
        """
        Create inertial shaker model. The default model is Dataphysics IV40

        Parameters
        ----------
        k : float, optional
            Spring constant. The default is 42992.
        m : float, optional
            Inertial mass. The default is 1.21.
        zeta : float, optional
            Damping ratio. The default is 0.05.
        Bl : float, optional
            Voice coil coefficient. The default is 5.6.
        R : float, optional
            Coil resistance. The default is 5.2.
        L : float, optional
            Coil inductance. The default is 0.298e-3.
        V_ext : float, optional
            Input voltage. The default is 1.

        Returns
        -------
        None.

        """

        c_c = 2*np.sqrt(k*m)
        self.c_viscous = c_c*zeta  # damping ratio

        self.k = k
        self.m = m
        self.Bl = Bl
        self.R = R
        self.L = L

        self.V_ext = V_ext

    def accelerance(self, f):
        """
        Calculate the acceleration response alpha_a(omega)

        Parameters
        ----------
        omega : float or array-like
            Angular frequency.

        Returns
        -------
        alpha_a : complex
            Acceleration response.
        """
        omega = 2*np.pi*f

        m_s = self.m
        c_s = self.c_viscous
        k_s = self.k

        numerator = -omega**2
        denominator = -omega**2 * m_s + 1j * omega * c_s + k_s

        alpha_a = numerator / denominator

        return alpha_a

    def electrical_impedance(self, f, alpha_0):
        """
        Calculate the voltage to current transfer function E(omega) / I(omega)

        Parameters
        ----------
        omega : float or array-like
            Angular frequency.
        alpha_0 : complex
            Reference acceleration response.

        Returns
        -------
        E_over_I : complex
            Voltage to current transfer function.
        """
        omega = 2*np.pi*f

        R = self.R
        L1 = self.L
        Bl = self.Bl

        alpha_a = self.accelerance(f)
        KfKv = Bl**2

        term1 = R + 1j * omega * L1
        term2 = (1 / (1j * omega)) * \
            (KfKv * (alpha_0 / (1 + (alpha_0 / alpha_a))))

        E_over_I = term1 + term2

        return E_over_I

    def force(self, f, u0):

        alpha_0 = (2*np.pi*1j*f)**2 * u0
        alpha_a = self.accelerance(f)

        fwfe_per_fshaker = alpha_a / \
            (alpha_a + alpha_0)

        Z_e = self.electrical_impedance(f, alpha_0)

        fshaker_per_vshaker = self.Bl * (1/Z_e)

        force = fwfe_per_fshaker*fshaker_per_vshaker * self.V_ext

        return force
