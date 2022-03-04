import numpy as np
import matplotlib.pyplot as plt
import os
from graphviz import Digraph # not necessary if you don't use 'visualize_model' command

class Model:
    def __init__(self, model_name, unit_mass = 'T', unit_time = 'yr'):
        self.name = model_name
        self.reservoirs = {0: {'box_name': 'external', 'M0': np.inf}}
        self.fluxes = {}
        self.unit = {'mass': unit_mass, 'time': unit_time}
        self.results = None
    def add_reservoir(self, number, box_name, M0 = 1):
        assert number not in self.reservoirs.keys(), f"key {number} is already used"
        self.reservoirs[number] = {'box_name': box_name, 'M0': M0}
    def add_flux(self, res_no_from, res_no_to, func, name = ''):
        assert res_no_from != res_no_to
        key = f"{res_no_from:0=2}_{res_no_to:0=2}"
        self.fluxes[key] = {'func': func, 'from': res_no_from, 
                            'to': res_no_to, 'name': name}
        
    def visualize_model(self):
        G = Digraph(format="png")
        G.attr("node", shape="box", width="1", color="orange")
        # display reservoirs
        for k, v in self.reservoirs.items():
            if k ==0:
                display_name = f"{k:0=2} {v['box_name']}"
                G.node(display_name, color = "black")
            else:
                display_name = f"{k:0=2} {v['box_name']}{os.linesep}M0 {v['M0']}"
                G.node(display_name)
            self.reservoirs[k]['display_name'] = display_name
        # display fluxes
        for k, v in self.fluxes.items():
            display_name_from = self.reservoirs[v['from']]['display_name']
            display_name_to = self.reservoirs[v['to']]['display_name']
            G.edge(display_name_from, display_name_to, label = f" {v['name']} ")
        G.render(f"charts/{self.name}")
        print(f"saved: charts/{self.name}.png")
    
    # main
    def run(self, t_init = 0, t_end = 10000, t_step = 1, show_progress = 10000):
        if self.results:
            reset_check = input('Results already exists. Reset results? (y for yes): ') == 'y'
            assert reset_check, 'save or generate new model'
        for k in self.fluxes.keys():
            self.fluxes[k]['hist'] = []
        t_list = list(range(t_init, t_end, t_step))
        Masses = np.zeros((len(t_list) + 1, len(self.reservoirs)))
        Masses[0] = np.array([v['M0'] for v in self.reservoirs.values()])
        self.masses = Masses[0]
        for i, t in enumerate(t_list):
            if (i + 1) % show_progress == 0:
                print(f"{i + 1} / {len(t_list)} calculated")
            d = dYdt(self, t)
            self.masses += d * t_step
            Masses[i + 1] = self.masses
        t_list += [t_list[-1] + t_step]
        self.results = {'t_list': t_list, 'Masses': Masses}
        
    def visualize_masses(self, key_number_list = None):
        assert self.results is not None, 'no results'
        if not key_number_list:
            key_number_list = [n for n in self.reservoirs.keys() if n != 0]
        for k in key_number_list:
            box_name = self.reservoirs[k]['box_name']
            t_list = self.results['t_list']
            mass = self.results['Masses'][:, k]
            plt.plot(t_list, mass)
            plt.title(f"reservoir: {box_name}")
            plt.xlabel(f"Time [{self.unit['time']}]")
            plt.ylabel(f"Mass [{self.unit['mass']}]")
            plt.show()
    
    def visualize_fluxes(self, key_list = None):
        assert self.results is not None, 'no results'
        if not key_list:
            key_list = list(self.fluxes.keys())
        for k in key_list:
            flux_name = self.fluxes[k]['name']
            tf_array = np.array(self.fluxes[k]['hist'])
            plt.plot(tf_array[:, 0], tf_array[:, 1])
            plt.title(f"flux: {flux_name}")
            plt.xlabel(f"Time [{self.unit['time']}]")
            plt.ylabel(f"Flux [{self.unit['mass']} / {self.unit['time']}]")
            plt.show()
            
# differential function
def dYdt(model, t):
    d = np.zeros(max(model.reservoirs.keys()) + 1)
    for v_flux in model.fluxes.values():
        res_from, res_to = v_flux['from'], v_flux['to']
        f = v_flux['func'](model, t, record = True)
        d[res_from] -= f
        d[res_to] += f
    return(d)