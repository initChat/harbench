"""
OPPORTUNITY (OPPORTUNITY Activity Recognition Dataset) Preprocessing

OPPORTUNITY Dataset:
- 17 types of mid-level gestures (+ Null class)
- 4 subjects
- 113 channels of body-worn sensors (7 IMUs + 12 accelerometer sensors)
- Sampling rate: 30Hz
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    create_sliding_windows_multi_session,
    filter_invalid_samples,
    get_class_distribution,
    resample_timeseries
)
from .common import (
    download_file,
    extract_archive,
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


# OPPORTUNITY dataset URL
OPPORTUNITY_URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"


# Select 113 sensor channels + 2 label columns from the 250-column raw data.
# Raw layout (0-indexed):
#   0         : timestamp (MILLISEC)
#   1-36      : 12 bluetooth ACC-only sensors (RKN^, HIP, LUA^, RUA_, LH, BACK_bt,
#               RKN_, RWR, RUA^, LUA_, LWR, RH) — 3 ch each
#   37-45     : IMU BACK (ACC[37-39], GYRO[40-42], MAG[43-45])
#   46-49     : IMU BACK QUAT  → removed
#   50-58     : IMU RUA  (ACC[50-52], GYRO[53-55], MAG[56-58])
#   59-62     : IMU RUA  QUAT  → removed
#   63-71     : IMU RLA  (ACC[63-65], GYRO[66-68], MAG[69-71])
#   72-75     : IMU RLA  QUAT  → removed
#   76-84     : IMU LUA  (ACC[76-78], GYRO[79-81], MAG[82-84])
#   85-88     : IMU LUA  QUAT  → removed
#   89-97     : IMU LLA  (ACC[89-91], GYRO[92-94], MAG[95-97])
#   98-101    : IMU LLA  QUAT  → removed
#   102-117   : L-SHOE 16 ch (Euler[102-104], Nav_Acc[105-107], Body_Acc[108-110],
#               AngVelBody[111-113], AngVelNav[114-116], Compass[117])
#   118-133   : R-SHOE 16 ch (same structure)
#   134-242   : object/ambient/location sensors → removed
#   243       : Locomotion label → removed
#   244       : HL_Activity label → removed
#   245       : LL_Left_Arm label → removed
#   246       : LL_Left_Arm_Object label → removed
#   247       : LL_Right_Arm label → removed
#   248       : LL_Right_Arm_Object label → removed
#   249       : ML_Both_Arms label (mid-level gestures) ← kept
#
# After deletion selected_data has 116 columns (0-indexed 0-115):
#   0         : timestamp
#   1-36      : bluetooth ACC sensors (sensor_data indices 0-35)
#   37-45     : IMU BACK without QUAT (sensor_data 36-44)
#   46-54     : IMU RUA  without QUAT (sensor_data 45-53)
#   55-63     : IMU RLA  without QUAT (sensor_data 54-62)
#   64-72     : IMU LUA  without QUAT (sensor_data 63-71)
#   73-81     : IMU LLA  without QUAT (sensor_data 72-80)
#   82-97     : L-SHOE 16 ch          (sensor_data 81-96)
#   98-113    : R-SHOE 16 ch          (sensor_data 97-112)
#   114       : Locomotion label
#   115       : ML_Both_Arms label    ← LABEL_COLUMN
def select_columns_opp(data):
    """
    Remove IMU quaternions, object/ambient sensors, and unused label columns.
    Keeps 113 sensor channels (selected_data[:,1:114]) + ML_Both_Arms label (selected_data[:,115]).

    Args:
        data: Original data matrix (samples, 250)

    Returns:
        selected_data: (samples, 116)
    """
    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    return np.delete(data, features_delete, 1)


# Sensor channel indices in sensor_data = selected_data[:, 1:114]  (0-indexed)
#
# Bluetooth ACC-only sensors (no GYRO/MAG):
#   RKN^  (Right Knee upper) : 0-2
#   HIP                      : 3-5
#   LUA^  (bluetooth)        : 6-8
#   RUA_  (bluetooth)        : 9-11
#   LH    (Left Hand)        : 12-14
#   BACK  (bluetooth)        : 15-17
#   RKN_  (Right Knee lower) : 18-20
#   RWR   (Right Wrist)      : 21-23
#   RUA^  (bluetooth)        : 24-26
#   LUA_  (bluetooth)        : 27-29
#   LWR   (Left Wrist)       : 30-32
#   RH    (Right Hand)       : 33-35
#
# IMU sensors (ACC + GYRO + MAG):
#   BACK : ACC[36-38]  GYRO[39-41]  MAG[42-44]
#   RUA  : ACC[45-47]  GYRO[48-50]  MAG[51-53]
#   RLA  : ACC[54-56]  GYRO[57-59]  MAG[60-62]
#   LUA  : ACC[63-65]  GYRO[66-68]  MAG[69-71]
#   LLA  : ACC[72-74]  GYRO[75-77]  MAG[78-80]
#
# Shoe sensors (no magnetometer; Body_Acc + AngVelBody used):
#   L_SHOE: Body_Acc[84-86]  AngVelBody[90-92]   (full 16-ch block: 81-96)
#   R_SHOE: Body_Acc[100-102] AngVelBody[106-108] (full 16-ch block: 97-112)
SENSOR_GROUPS = {
    'BACK': {
        'channels': list(range(36, 45)),
        'modalities': {
            'ACC': list(range(36, 39)),
            'GYRO': list(range(39, 42)),
            'MAG': list(range(42, 45)),
        }
    },
    'RUA': {  # Right Upper Arm IMU
        'channels': list(range(45, 54)),
        'modalities': {
            'ACC': list(range(45, 48)),
            'GYRO': list(range(48, 51)),
            'MAG': list(range(51, 54)),
        }
    },
    'RLA': {  # Right Lower Arm IMU
        'channels': list(range(54, 63)),
        'modalities': {
            'ACC': list(range(54, 57)),
            'GYRO': list(range(57, 60)),
            'MAG': list(range(60, 63)),
        }
    },
    'LUA': {  # Left Upper Arm IMU
        'channels': list(range(63, 72)),
        'modalities': {
            'ACC': list(range(63, 66)),
            'GYRO': list(range(66, 69)),
            'MAG': list(range(69, 72)),
        }
    },
    'LLA': {  # Left Lower Arm IMU
        'channels': list(range(72, 81)),
        'modalities': {
            'ACC': list(range(72, 75)),
            'GYRO': list(range(75, 78)),
            'MAG': list(range(78, 81)),
        }
    },
    'L_SHOE': {  # Left Shoe (InertiaCube3) — no magnetometer
        'channels': list(range(81, 97)),
        'modalities': {
            'ACC': list(range(84, 87)),   # Body_Acc XYZ
            'GYRO': list(range(90, 93)),  # AngVelBody XYZ
        }
    },
    'R_SHOE': {  # Right Shoe (InertiaCube3) — no magnetometer
        'channels': list(range(97, 113)),
        'modalities': {
            'ACC': list(range(100, 103)),  # Body_Acc XYZ
            'GYRO': list(range(106, 109)), # AngVelBody XYZ
        }
    },
    'R_WRIST': {  # Right Wrist bluetooth accelerometer (ACC only)
        'channels': list(range(21, 24)),
        'modalities': {
            'ACC': list(range(21, 24)),
        }
    },
    'R_KNEE': {  # Right Knee (RKN^) bluetooth accelerometer (ACC only)
        'channels': list(range(0, 3)),
        'modalities': {
            'ACC': list(range(0, 3)),
        }
    },
}

# Label column index in selected_data (ML_Both_Arms = mid-level gestures)
LABEL_COLUMN = 115


@register_preprocessor('opportunity')
class OpportunityPreprocessor(BasePreprocessor):
    """
    Preprocessing class for OPPORTUNITY dataset (using all 113 sensor channels)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # OPPORTUNITY-specific settings
        self.num_activities = 17  # mid-level gestures
        self.num_subjects = 4
        self.num_channels = 113  # All body-worn sensors

        # Sampling rates
        self.original_sampling_rate = 30  # Hz (OPPORTUNITY original)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (target)

        # Sensor groups
        self.sensor_groups = SENSOR_GROUPS
        self.sensor_names = list(SENSOR_GROUPS.keys())

        # Preprocessing parameters
        self.window_size = config.get('window_size', 30)  # 1 second @ 30Hz
        self.stride = config.get('stride', 15)  # 0.5 second @ 30Hz (50% overlap)

        # Scaling factor (m/s^2 -> G conversion)
        self.scale_factor = DATASETS.get('OPPORTUNITY', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'opportunity'

    def download_dataset(self) -> None:
        """
        Download and extract OPPORTUNITY dataset
        """
        logger.info("=" * 80)
        logger.info("Downloading OPPORTUNITY dataset")
        logger.info("=" * 80)

        opportunity_raw_path = self.raw_data_path / self.dataset_name

        # Check if data already exists
        if check_dataset_exists(opportunity_raw_path, required_files=['*.dat']):
            logger.warning(f"OPPORTUNITY data already exists at {opportunity_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. Download
            logger.info("Step 1/2: Downloading archive")
            opportunity_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = opportunity_raw_path.parent / 'opportunity.zip'
            download_file(OPPORTUNITY_URL, zip_path, desc='Downloading OPPORTUNITY')

            # 2. Extract and organize data
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = opportunity_raw_path.parent / 'opportunity_temp'
            extract_archive(zip_path, extract_to, desc='Extracting OPPORTUNITY')
            self._organize_opportunity_data(extract_to, opportunity_raw_path)

            # Cleanup
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: OPPORTUNITY dataset downloaded to {opportunity_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download OPPORTUNITY dataset: {e}", exc_info=True)
            raise

    def _organize_opportunity_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        Organize OPPORTUNITY data into proper directory structure

        Args:
            extracted_path: Path to extracted data
            target_path: Path to save organized data (data/raw/opportunity)
        """
        import shutil
        from tqdm import tqdm

        logger.info(f"Organizing OPPORTUNITY data from {extracted_path} to {target_path}")

        # Find the root of extracted data
        data_root = extracted_path

        # Look for "OpportunityUCIDataset" folder
        if (extracted_path / "OpportunityUCIDataset").exists():
            data_root = extracted_path / "OpportunityUCIDataset"
            if (data_root / "dataset").exists():
                data_root = data_root / "dataset"

        # Create target directory
        target_path.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Could not find data directory in {extracted_path}")

        # Find and copy .dat files
        dat_files = list(data_root.glob("*.dat"))

        if not dat_files:
            # Search in subdirectories as well
            dat_files = list(data_root.glob("**/*.dat"))

        if not dat_files:
            raise FileNotFoundError(f"No .dat files found in {data_root}")

        logger.info(f"Found {len(dat_files)} .dat files")

        for dat_file in tqdm(dat_files, desc='Organizing files'):
            target_file = target_path / dat_file.name
            if target_file.exists():
                target_file.unlink()
            shutil.copy2(dat_file, target_file)

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, list]:
        """
        Load OPPORTUNITY raw data by subject

        Expected format:
        - data/raw/opportunity/S1-ADL1.dat, S1-ADL2.dat, ..., S1-Drill.dat
        - Each file: (samples, 249) text file (space-delimited)

        Returns:
            person_data: Dictionary {person_id: [(data, labels), ...]}
                Each element is per file (session)
                data: Array of (num_samples, 113) (selected sensor columns)
                labels: Array of (num_samples,)
        """
        # raw_path = self.raw_data_path / self.dataset_name
        raw_path = self.raw_data_path / self.dataset_name / "dataset"

        if not raw_path.exists():
            raise FileNotFoundError(
                f"OPPORTUNITY raw data not found at {raw_path}\n"
                "Expected structure: data/raw/opportunity/S1-ADL1.dat"
            )

        # Store data for each subject
        person_data = {person_id: {'sessions': []}
                       for person_id in range(1, self.num_subjects + 1)}

        # Load files for each subject
        for person_id in range(1, self.num_subjects + 1):
            # Load ADL files and Drill file
            file_patterns = [
                f"S{person_id}-ADL*.dat",
                f"S{person_id}-Drill.dat"
            ]

            subject_files = []
            for pattern in file_patterns:
                subject_files.extend(sorted(raw_path.glob(pattern)))

            if not subject_files:
                logger.warning(f"No data files found for subject S{person_id}")
                continue

            logger.info(f"Loading {len(subject_files)} files for USER{person_id:05d}")

            for data_file in subject_files:
                try:
                    # Load data (space-delimited)
                    data = np.loadtxt(data_file, dtype=np.float32)

                    if data.ndim == 1:
                        data = data.reshape(1, -1)

                    logger.info(f"  Loaded {data_file.name}: {data.shape}")

                    # Column selection (extract 113 channels)
                    selected_data = select_columns_opp(data)

                    # ML_Both_Arms (mid-level gesture) is at selected_data[:,115]
                    # selected_data[:,114] is Locomotion — do not use
                    labels = selected_data[:, 115].astype(np.int32)

                    # Sensor data: 113 channels (columns 1-113 of selected_data)
                    sensor_data = selected_data[:, 1:114]

                    # Label conversion: 0 -> -1 (Null class), others adjusted
                    # gestures label adjustment (based on DeepConvLSTM implementation)
                    label_map = {
                        0: -1,       # Null -> -1
                        406516: 0,   # Open Door 1
                        406517: 1,   # Open Door 2
                        404516: 2,   # Close Door 1
                        404517: 3,   # Close Door 2
                        406520: 4,   # Open Fridge
                        404520: 5,   # Close Fridge
                        406505: 6,   # Open Dishwasher
                        404505: 7,   # Close Dishwasher
                        406519: 8,   # Open Drawer 1
                        404519: 9,   # Close Drawer 1
                        406511: 10,  # Open Drawer 2
                        404511: 11,  # Close Drawer 2
                        406508: 12,  # Open Drawer 3
                        404508: 13,  # Close Drawer 3
                        408512: 14,  # Clean Table
                        407521: 15,  # Drink from Cup
                        405506: 16,  # Toggle Switch
                    }

                    for old_label, new_label in label_map.items():
                        labels[labels == old_label] = new_label

                    person_data[person_id]['sessions'].append((sensor_data, labels))

                except Exception as e:
                    logger.error(f"Error loading {data_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Return session list for each subject (do not concatenate)
        result = {}
        for person_id in range(1, self.num_subjects + 1):
            if person_data[person_id]['sessions']:
                sessions = person_data[person_id]['sessions']
                result[person_id] = sessions
                total_samples = sum(d.shape[0] for d, l in sessions)
                logger.info(f"USER{person_id:05d}: {len(sessions)} sessions, {total_samples} samples")
            else:
                logger.warning(f"No data loaded for USER{person_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, list]) -> Dict[int, list]:
        """
        Data cleaning (process per session)

        Args:
            data: Dictionary {person_id: [(data, labels), ...]}

        Returns:
            Cleaned {person_id: [(data, labels), ...]}
        """
        cleaned = {}
        for person_id, sessions in data.items():
            cleaned_sessions = []
            for session_data, session_labels in sessions:
                # Remove rows containing NaN/Inf
                cleaned_data, cleaned_labels = filter_invalid_samples(session_data, session_labels)

                if len(cleaned_data) == 0:
                    continue

                # Sampling rate is already 30Hz, so no resampling needed
                cleaned_sessions.append((cleaned_data, cleaned_labels))

            if cleaned_sessions:
                cleaned[person_id] = cleaned_sessions
                total_samples = sum(d.shape[0] for d, l in cleaned_sessions)
                logger.info(f"USER{person_id:05d} cleaned: {len(cleaned_sessions)} sessions, {total_samples} samples")

        return cleaned

    def extract_features(self, data: Dict[int, list]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Feature extraction (windowing and scaling per sensor group × modality)
        Do not generate windows that cross session boundaries

        Args:
            data: Dictionary {person_id: [(data, labels), ...]}

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for person_id, sessions in data.items():
            logger.info(f"Processing USER{person_id:05d} ({len(sessions)} sessions)")

            # Skip if sessions are empty
            if len(sessions) == 0:
                logger.warning(f"  USER{person_id:05d} has no valid sessions, skipping")
                continue

            processed[person_id] = {}

            # Process each sensor group
            for sensor_name, sensor_info in self.sensor_groups.items():
                for modality_name, modality_channels in sensor_info['modalities'].items():
                    # Extract channels for this modality per session
                    modality_sessions = [
                        (session_data[:, modality_channels], session_labels)
                        for session_data, session_labels in sessions
                    ]

                    # Apply sliding window per session
                    windowed_data, windowed_labels = create_sliding_windows_multi_session(
                        modality_sessions,
                        window_size=self.window_size,
                        stride=self.stride,
                        drop_last=False,
                        pad_last=True
                    )

                    if len(windowed_data) == 0:
                        logger.warning(f"  {sensor_name}/{modality_name}: no valid windows")
                        continue

                    # Apply scaling (accelerometer only)
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        windowed_data = windowed_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # Reshape: (num_windows, window_size, channels) -> (num_windows, channels, window_size)
                    windowed_data = np.transpose(windowed_data, (0, 2, 1))

                    # Convert to float16
                    windowed_data = windowed_data.astype(np.float16)

                    # Sensor/modality hierarchical structure
                    sensor_modality_key = f"{sensor_name}/{modality_name}"

                    processed[person_id][sensor_modality_key] = {
                        'X': windowed_data,
                        'Y': windowed_labels
                    }

                    logger.info(
                        f"  {sensor_modality_key}: X.shape={windowed_data.shape}, "
                        f"Y.shape={windowed_labels.shape}"
                    )

        return processed

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        Save processed data

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        Save format:
            data/processed/opportunity/USER00001/BACK/ACC/X.npy, Y.npy
            data/processed/opportunity/USER00001/BACK/GYRO/X.npy, Y.npy
            ...
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_channels': self.num_channels,
            'sensor_groups': list(self.sensor_groups.keys()),
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # No normalization (keep raw data)
            'scale_factor': self.scale_factor,  # Scaling factor (applied to ACC only)
            'data_dtype': 'float16',  # Data type
            'users': {}
        }

        for person_id, sensor_modality_data in data.items():
            user_name = f"USER{person_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                # Save X.npy, Y.npy
                X = arrays['X']  # (num_windows, channels, window_size)
                Y = arrays['Y']  # (num_windows,)

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                # Statistics
                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': X.shape,
                    'Y_shape': Y.shape,
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(
                    f"Saved {user_name}/{sensor_modality_name}: "
                    f"X{X.shape}, Y{Y.shape}"
                )

            total_stats['users'][user_name] = user_stats

        # Save overall metadata
        metadata_path = base_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            # Convert NumPy types to JSON-compatible format
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, tuple):
                    return list(obj)
                return obj

            def recursive_convert(d):
                if isinstance(d, dict):
                    return {k: recursive_convert(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [recursive_convert(v) for v in d]
                else:
                    return convert_to_serializable(d)

            serializable_stats = recursive_convert(total_stats)
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")
