"""Functions for contacts."""
from .analyse import analyse_channels_in_trc, analyse_channels_in_mat  # noqa
from .utils import (clean_contact, contact_mono_to_bipo, contact_bipo_to_mono,  # noqa
                    successive_monopolar_contacts, compute_middle_contact,
                    contact_to_mni, detect_seeg_contacts)