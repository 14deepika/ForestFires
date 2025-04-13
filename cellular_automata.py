# cellular_automata.py

import numpy as np
import joblib

class CellularAutomataFire:
    """
    A simple Cellular Automata for forest fire spread.
    Grid states: 0 = unburned, 1 = burning, 2 = burned.
    The ML model is used to determine ignition probability
    for an unburned cell if any neighbor is burning.
    """

    def __init__(self, rows, cols, model_path="forest_fire_model.pkl"):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)  # 0=unburned,1=burning,2=burned
        self.model = joblib.load(model_path)

        # Initialize a burning cell in the center
        self.grid[rows // 2, cols // 2] = 1

        # Features that the model expects
        self.feature_cols = [
            "X","Y","month","day","FFMC","DMC","DC","ISI","temp","RH","wind","rain"
        ]

        # Default environment data (overridden by user in /simulate)
        self.env_data = {
            "month": 8,   # August
            "day": 15,
            "FFMC": 90.0,
            "DMC": 35.0,
            "DC": 100.0,
            "ISI": 5.0,
            "temp": 20.0,
            "RH": 40,
            "wind": 3.0,
            "rain": 0.0
        }

    def get_neighbors(self, r, c):
        neighbors = []
        for nr in [r-1, r, r+1]:
            for nc in [c-1, c, c+1]:
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if not (nr == r and nc == c):
                        neighbors.append((nr, nc))
        return neighbors

    def step(self):
        new_grid = self.grid.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 0:
                    # If unburned, check if any neighbor is burning
                    neighbors = self.get_neighbors(r, c)
                    if any(self.grid[nr, nc] == 1 for nr, nc in neighbors):
                        # Build features for the ML model
                        sample_data = {"X": r, "Y": c}
                        sample_data.update(self.env_data)
                        input_list = [sample_data[col] for col in self.feature_cols]
                        input_array = np.array(input_list).reshape(1, -1)

                        # Probability that this cell ignites
                        prob = self.model.predict_proba(input_array)[0][1]
                        if prob > 0.5:
                            new_grid[r, c] = 1
                elif self.grid[r, c] == 1:
                    # Burning cell becomes burned
                    new_grid[r, c] = 2
        self.grid = new_grid

    def run_simulation(self, steps=10):
        for _ in range(steps):
            self.step()
        return self.grid
