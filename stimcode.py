import stim
from pymatching import Matching
import numpy as np

p = 0.007
d = 11
r = 9

circuit = stim.Circuit.generated("surface_code:rotated_memory_z", 
                                    distance=d, 
                                    rounds=r, 
                                    after_clifford_depolarization=p,
                                    before_round_data_depolarization=p,
                                    after_reset_flip_probability=p,
                                    before_measure_flip_probability=p)
                                        
model = circuit.detector_error_model(decompose_errors=True)
matching = Matching.from_detector_error_model(model)

sampler = circuit.compile_detector_sampler()

def save_bool_array(path, arr):
    """
    Save a large boolean array efficiently.
    Stores data as packed bits.
    """
    arr = np.asarray(arr, dtype=np.bool_)
    packed = np.packbits(arr, axis=1)
    packed.tofile(path)
    
def load_bool_array(path, shape):
    """
    Load packed boolean array using memory mapping.
    """
    rows, cols = shape
    packed_cols = (cols + 7) // 8
    mm = np.memmap(path,
        dtype=np.uint8,
        mode="r",
        shape=(rows, packed_cols),)
    arr = np.unpackbits(mm, axis=1)
    return arr[:, :cols].astype(bool, copy=False)

n_shots = 1000000
N = 300
for k in range(N):
    syndrome, actual_observables = sampler.sample(shots=n_shots, separate_observables=True)
    predicted_observables = matching.decode_batch(syndrome)
    save_bool_array("syndrome-"+str(k)+".npb", syndrome)
    save_bool_array("obs-"+str(k)+".npb", (actual_observables!=predicted_observables))
