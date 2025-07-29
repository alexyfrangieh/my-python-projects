#!/usr/bin/env python3
"""
EXIF to Database Extractor - Fresh Start
Extracts EXIF data from images using exiftool and stores in MySQL database
Usage: python exif2db.py /path/to/images
"""

import os
import sys
import json
import csv
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import pymysql
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import time

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Aadmin9_',
    'database': 'sohdo',
    'charset': 'utf8mb4'
}

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.raw', '.cr2', '.nef', '.arw'}

# --- Helper functions for multiprocessing (outside the class) ---
def _safe_convert_int_worker(value):
    """Safely convert value to int, return None if fails"""
    if value is None:
        return None
    try:
        str_val = str(value).strip()
        numbers = re.findall(r'\d+', str_val)
        if numbers:
            return int(numbers[0])
        return None
    except:
        return None

def _parse_gps_coord_worker(coord_str):
    """Parse GPS coordinates from various formats"""
    if not coord_str:
        return None
    try:
        if isinstance(coord_str, (int, float)):
            return float(coord_str)
        coord_str = str(coord_str)
        if 'deg' in coord_str or "'" in coord_str:
            numbers = re.findall(r'[\d.]+', coord_str)
            if len(numbers) >= 3:
                deg, min_val, sec = map(float, numbers[:3])
                decimal = deg + min_val/60 + sec/3600
                if 'S' in coord_str or 'W' in coord_str:
                    decimal = -decimal
                return decimal
        return float(coord_str)
    except:
        return None

def _parse_datetime_worker(date_str):
    """Parse datetime with multiple format support, removing timezone info"""
    if not date_str:
        return None
    try:
        date_str = str(date_str)
        if '+' in date_str:
            date_str = date_str.split('+')[0]
        if '-' in date_str and len(date_str) > 19:
            date_str = date_str[:19]

        for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        return date_str
    except:
        return str(date_str) if date_str else None

def _extract_and_parse_exif_worker(image_path_str):
    """
    Worker function executed by the ProcessPoolExecutor.
    Extracts EXIF data using exiftool and parses it into a structured dictionary.
    """
    image_path = Path(image_path_str)
    worker_logger = logging.getLogger(f'worker_{os.getpid()}') # Corrected: os.getpid()
    if not worker_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('exif_extraction.log', mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )

    try:
        cmd = ['exiftool', '-json', '-charset', 'utf8', str(image_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            worker_logger.warning(f"⚠ exiftool failed for {image_path}: {result.stderr.strip()}")
            return None

        exif_raw = json.loads(result.stdout)[0]

        def safe_get(key, default=None):
            return exif_raw.get(key, default)

        parsed_data = {
            'source_file': str(image_path),
            'file_name': image_path.name,
            'directory': str(safe_get('Directory', '')),
            'file_size': str(safe_get('FileSize', '')),
            'file_modify_date': str(safe_get('FileModifyDate', '')),
            'file_type': str(safe_get('FileType', '')),
            'mime_type': str(safe_get('MIMEType', '')),
            'make': str(safe_get('Make', '')),
            'model': str(safe_get('Model', '')),
            'serial_number': str(safe_get('SerialNumber', '')),
            'firmware_version': str(safe_get('FirmwareVersion', '')),
            'software': str(safe_get('Software', '')),
            'image_width': _safe_convert_int_worker(safe_get('ImageWidth') or safe_get('ExifImageWidth')),
            'image_height': _safe_convert_int_worker(safe_get('ImageHeight') or safe_get('ExifImageHeight')),
            'orientation': str(safe_get('Orientation', '')),
            'resolution_unit': str(safe_get('ResolutionUnit', '')),
            'x_resolution': str(safe_get('XResolution', '')),
            'y_resolution': str(safe_get('YResolution', '')),
            'color_space': str(safe_get('ColorSpace', '')),
            'exposure_time': str(safe_get('ExposureTime', '')),
            'f_number': str(safe_get('FNumber', '')),
            'aperture': str(safe_get('Aperture', '')),
            'iso_speed': str(safe_get('ISO', '')),
            'iso_setting': str(safe_get('ISOSetting', '')),
            'exposure_program': str(safe_get('ExposureProgram', '')),
            'exposure_mode': str(safe_get('ExposureMode', '')),
            'exposure_compensation': str(safe_get('ExposureCompensation', '')),
            'metering_mode': str(safe_get('MeteringMode', '')),
            'date_time_original': _parse_datetime_worker(safe_get('DateTimeOriginal')),
            'create_date': _parse_datetime_worker(safe_get('CreateDate')),
            'modify_date': _parse_datetime_worker(safe_get('ModifyDate')),
            'sub_sec_time': str(safe_get('SubSecTime', '')),
            'sub_sec_create_date': str(safe_get('SubSecCreateDate', '')),
            'flash': str(safe_get('Flash', '')),
            'flash_mode': str(safe_get('FlashMode', '')),
            'flash_setting': str(safe_get('FlashSetting', '')),
            'flash_type': str(safe_get('FlashType', '')),
            'flash_compensation': str(safe_get('FlashCompensation', '')),
            'flash_exposure_comp': str(safe_get('FlashExposureComp', '')),
            'focus_mode': str(safe_get('FocusMode', '')),
            'af_area_mode': str(safe_get('AFAreaMode', '')),
            'focus_distance': str(safe_get('FocusDistance', '')),
            'focus_position': str(safe_get('FocusPosition', '')),
            'lens': str(safe_get('Lens', '')),
            'lens_type': str(safe_get('LensType', '')),
            'lens_id': str(safe_get('LensID', '')),
            'focal_length': str(safe_get('FocalLength', '')),
            'focal_length_35mm': str(safe_get('FocalLengthIn35mmFormat', '')),
            'min_focal_length': str(safe_get('MinFocalLength', '')),
            'max_focal_length': str(safe_get('MaxFocalLength', '')),
            'max_aperture_value': str(safe_get('MaxApertureValue', '')),
            'white_balance': str(safe_get('WhiteBalance', '')),
            'white_balance_fine_tune': str(safe_get('WhiteBalanceFineTune', '')),
            'wb_rb_levels': str(safe_get('WB_RBLevels', '')),
            'wb_rggb_levels': str(safe_get('WB_RGGBLevels', '')),
            'quality': str(safe_get('Quality', '')),
            'shooting_mode': str(safe_get('ShootingMode', '')),
            'picture_control_name': str(safe_get('PictureControlName', '')),
            'picture_control_base': str(safe_get('PictureControlBase', '')),
            'contrast': str(safe_get('Contrast', '')),
            'saturation': str(safe_get('Saturation', '')),
            'sharpness': str(safe_get('Sharpness', '')),
            'brightness': str(safe_get('Brightness', '')),
            'gps_latitude': _parse_gps_coord_worker(safe_get('GPSLatitude')),
            'gps_longitude': _parse_gps_coord_worker(safe_get('GPSLongitude')),
            'gps_altitude': _parse_gps_coord_worker(safe_get('GPSAltitude')),
            'shutter_count': _safe_convert_int_worker(safe_get('ShutterCount')),
            'light_value': str(safe_get('LightValue', '')),
            'noise_reduction': str(safe_get('NoiseReduction', '')),
            'vibration_reduction': str(safe_get('VibrationReduction', '')),
            'active_d_lighting': str(safe_get('ActiveD-Lighting', '')),
            'scene_capture_type': str(safe_get('SceneCaptureType', '')),
            'exif_raw': json.dumps(exif_raw, indent=2),
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        worker_logger.debug(f"✓ Extracted and parsed EXIF for {image_path.name}")
        return parsed_data

    except subprocess.TimeoutExpired:
        worker_logger.error(f"✗ Timeout extracting EXIF from {image_path}")
        return None
    except json.JSONDecodeError as jde:
        worker_logger.error(f"✗ JSON decode error for {image_path}: {jde}. Raw output: {result.stdout}")
        return None
    except Exception as e:
        worker_logger.error(f"✗ EXIF extraction/parsing failed for {image_path}: {e}")
        return None

# --- End of helper functions ---

class ExifExtractor:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.connection = None
        self.csv_output_path = None
        self.processed_count = 0
        self.error_count = 0
        self.batch_size = 50
        self._data_buffer = []  # Accumulates ALL parsed data for final CSV
        self.start_time = None
        self.total_images = 0

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('exif_extraction.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _display_progress(self):
        if self.start_time is None:
            return

        time_elapsed = time.time() - self.start_time
        avg_time_per_image = time_elapsed / self.processed_count if self.processed_count > 0 else 0

        hours, remainder = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_elapsed_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        avg_time_str = f"{avg_time_per_image:.2f}s" if avg_time_per_image > 0 else "N/A"

        progress_str = (
            f"\rImages To Process: {self.total_images} | "
            f"Images Processed: {self.processed_count} | "
            f"Time Elapsed: {time_elapsed_str} | "
            f"Avg Time/Image: {avg_time_str}"
        )
        sys.stdout.write(progress_str.ljust(os.get_terminal_size().columns))
        sys.stdout.flush()

    def connect_db(self):
        try:
            self.connection = pymysql.connect(**DB_CONFIG)
            self.logger.info("✓ Connected to MySQL database")
            self.create_table()
        except Exception as e:
            self.logger.error(f"✗ Database connection failed: {e}")
            sys.exit(1)

    def create_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS exif (
            id INT AUTO_INCREMENT PRIMARY KEY,
            -- File Information
            source_file TEXT,
            file_name VARCHAR(255),
            directory TEXT,
            file_size VARCHAR(50),
            file_modify_date VARCHAR(100),
            file_type VARCHAR(50),
            mime_type VARCHAR(100),
            -- Camera Information
            make VARCHAR(100),
            model VARCHAR(100),
            serial_number VARCHAR(50),
            firmware_version VARCHAR(50),
            software VARCHAR(100),
            -- Image Settings
            image_width INT,
            image_height INT,
            orientation VARCHAR(50),
            resolution_unit VARCHAR(20),
            x_resolution VARCHAR(20),
            y_resolution VARCHAR(20),
            color_space VARCHAR(50),
            -- Exposure Settings
            exposure_time VARCHAR(50),
            f_number VARCHAR(20),
            aperture VARCHAR(20),
            iso_speed VARCHAR(20),
            iso_setting VARCHAR(20),
            exposure_program VARCHAR(100),
            exposure_mode VARCHAR(50),
            exposure_compensation VARCHAR(20),
            metering_mode VARCHAR(50),
            -- Date/Time
            date_time_original VARCHAR(100),
            create_date VARCHAR(100),
            modify_date VARCHAR(100),
            sub_sec_time VARCHAR(10),
            sub_sec_create_date VARCHAR(100),
            -- Flash Settings
            flash TEXT,
            flash_mode VARCHAR(50),
            flash_setting VARCHAR(50),
            flash_type VARCHAR(50),
            flash_compensation VARCHAR(20),
            flash_exposure_comp VARCHAR(20),
            -- Focus Settings
            focus_mode VARCHAR(50),
            af_area_mode VARCHAR(50),
            focus_distance VARCHAR(50),
            focus_position VARCHAR(50),
            -- Lens Information
            lens VARCHAR(200),
            lens_type VARCHAR(100),
            lens_id VARCHAR(50),
            focal_length VARCHAR(50),
            focal_length_35mm VARCHAR(50),
            min_focal_length VARCHAR(50),
            max_focal_length VARCHAR(50),
            max_aperture_value VARCHAR(20),
            -- White Balance
            white_balance VARCHAR(50),
            white_balance_fine_tune VARCHAR(50),
            wb_rb_levels VARCHAR(100),
            wb_rggb_levels VARCHAR(200),
            -- Picture Settings
            quality VARCHAR(50),
            shooting_mode VARCHAR(50),
            picture_control_name VARCHAR(50),
            picture_control_base VARCHAR(50),
            contrast VARCHAR(50),
            saturation VARCHAR(50),
            sharpness VARCHAR(50),
            brightness VARCHAR(50),
            -- GPS Data
            gps_latitude DECIMAL(10, 8),
            gps_longitude DECIMAL(11, 8),
            gps_altitude DECIMAL(8, 3),
            -- Technical Details
            shutter_count INT,
            light_value VARCHAR(20),
            noise_reduction VARCHAR(50),
            vibration_reduction VARCHAR(50),
            active_d_lighting VARCHAR(50),
            scene_capture_type VARCHAR(50),
            -- Raw EXIF Data
            exif_raw LONGTEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            -- Indexes
            INDEX idx_filename (file_name),
            INDEX idx_camera (make, model),
            INDEX idx_date (date_time_original),
            INDEX idx_serial (serial_number),
            INDEX idx_focal (focal_length),
            INDEX idx_iso (iso_speed)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS exif")
                cursor.execute(create_table_sql)
                self.connection.commit()
            self.logger.info("✓ Fresh EXIF table created successfully")
        except Exception as e:
            self.logger.error(f"✗ Table creation failed: {e}")
            sys.exit(1)

    def setup_csv(self):
        try:
            csv_path = self.base_path / 'exif_data.csv'
            with open(csv_path, 'w') as f:
                pass
            self.csv_output_path = csv_path
            self.logger.info(f"✓ Determined CSV output path: {self.csv_output_path}")
        except PermissionError:
            csv_path = Path.cwd() / 'exif_data.csv'
            try:
                with open(csv_path, 'w') as f:
                    pass
                self.csv_output_path = csv_path
                self.logger.info(f"✓ Determined CSV output path (fallback): {self.csv_output_path}")
            except Exception as e:
                self.logger.error(f"✗ Cannot determine a writable CSV path even with fallback: {e}")
                self.csv_output_path = None
        except Exception as e:
            self.logger.error(f"✗ Failed to determine CSV path: {e}")
            self.csv_output_path = None

    def _insert_batch_to_db(self, data_batch):
        if not data_batch:
            return True

        insert_sql = """
        INSERT INTO exif (
            source_file, file_name, directory, file_size, file_modify_date, file_type, mime_type,
            make, model, serial_number, firmware_version, software,
            image_width, image_height, orientation, resolution_unit, x_resolution, y_resolution, color_space,
            exposure_time, f_number, aperture, iso_speed, iso_setting, exposure_program, exposure_mode,
            exposure_compensation, metering_mode, date_time_original, create_date, modify_date,
            sub_sec_time, sub_sec_create_date, flash, flash_mode, flash_setting, flash_type,
            flash_compensation, flash_exposure_comp, focus_mode, af_area_mode, focus_distance, focus_position,
            lens, lens_type, lens_id, focal_length, focal_length_35mm, min_focal_length, max_focal_length,
            max_aperture_value, white_balance, white_balance_fine_tune, wb_rb_levels, wb_rggb_levels,
            quality, shooting_mode, picture_control_name, picture_control_base, contrast, saturation,
            sharpness, brightness, gps_latitude, gps_longitude, gps_altitude, shutter_count,
            light_value, noise_reduction, vibration_reduction, active_d_lighting, scene_capture_type,
            exif_raw, processed_at
        ) VALUES (
            %(source_file)s, %(file_name)s, %(directory)s, %(file_size)s, %(file_modify_date)s, %(file_type)s, %(mime_type)s,
            %(make)s, %(model)s, %(serial_number)s, %(firmware_version)s, %(software)s,
            %(image_width)s, %(image_height)s, %(orientation)s, %(resolution_unit)s, %(x_resolution)s, %(y_resolution)s, %(color_space)s,
            %(exposure_time)s, %(f_number)s, %(aperture)s, %(iso_speed)s, %(iso_setting)s, %(exposure_program)s, %(exposure_mode)s,
            %(exposure_compensation)s, %(metering_mode)s, %(date_time_original)s, %(create_date)s, %(modify_date)s,
            %(sub_sec_time)s, %(sub_sec_create_date)s, %(flash)s, %(flash_mode)s, %(flash_setting)s, %(flash_type)s,
            %(flash_compensation)s, %(flash_exposure_comp)s, %(focus_mode)s, %(af_area_mode)s, %(focus_distance)s, %(focus_position)s,
            %(lens)s, %(lens_type)s, %(lens_id)s, %(focal_length)s, %(focal_length_35mm)s, %(min_focal_length)s, %(max_focal_length)s,
            %(max_aperture_value)s, %(white_balance)s, %(white_balance_fine_tune)s, %(wb_rb_levels)s, %(wb_rggb_levels)s,
            %(quality)s, %(shooting_mode)s, %(picture_control_name)s, %(picture_control_base)s, %(contrast)s, %(saturation)s,
            %(sharpness)s, %(brightness)s, %(gps_latitude)s, %(gps_longitude)s, %(gps_altitude)s, %(shutter_count)s,
            %(light_value)s, %(noise_reduction)s, %(vibration_reduction)s, %(active_d_lighting)s, %(scene_capture_type)s,
            %(exif_raw)s, %(processed_at)s
        )
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.executemany(insert_sql, data_batch)
            self.connection.commit()
            self.logger.info(f"✓ Inserted batch of {len(data_batch)} records into DB.")
            return True
        except Exception as e:
            self.logger.error(f"✗ DB batch insertion failed. First file in batch: {data_batch[0].get('file_name', 'N/A')}. Error: {e}")
            return False

    def _write_all_to_csv(self):
        if not self.csv_output_path:
            self.logger.warning("No CSV output path defined. Skipping CSV export.")
            return

        if not self._data_buffer:
            self.logger.info("No data collected for CSV export.")
            return

        fieldnames = [
            'source_file', 'file_name', 'directory', 'file_size', 'file_modify_date',
            'file_type', 'mime_type', 'make', 'model', 'serial_number', 'firmware_version',
            'software', 'image_width', 'image_height', 'orientation', 'resolution_unit',
            'x_resolution', 'y_resolution', 'color_space', 'exposure_time', 'f_number',
            'aperture', 'iso_speed', 'iso_setting', 'exposure_program', 'exposure_mode',
            'exposure_compensation', 'metering_mode', 'date_time_original', 'create_date',
            'modify_date', 'sub_sec_time', 'sub_sec_create_date', 'flash', 'flash_mode',
            'flash_setting', 'flash_type', 'flash_compensation', 'flash_exposure_comp',
            'focus_mode', 'af_area_mode', 'focus_distance', 'focus_position', 'lens',
            'lens_type', 'lens_id', 'focal_length', 'focal_length_35mm', 'min_focal_length',
            'max_focal_length', 'max_aperture_value', 'white_balance', 'white_balance_fine_tune',
            'wb_rb_levels', 'wb_rggb_levels', 'quality', 'shooting_mode', 'picture_control_name',
            'picture_control_base', 'contrast', 'saturation', 'sharpness', 'brightness',
            'gps_latitude', 'gps_longitude', 'gps_altitude', 'shutter_count', 'light_value',
            'noise_reduction', 'vibration_reduction', 'active_d_lighting', 'scene_capture_type',
            'processed_at'
        ]

        try:
            with open(self.csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()

                csv_records = []
                for data in self._data_buffer:
                    csv_record = {k: (v if v is not None else '') for k, v in data.items() if k != 'exif_raw'}
                    csv_records.append(csv_record)
                csv_writer.writerows(csv_records)
            self.logger.info(f"✓ All collected EXIF data written to CSV: {self.csv_output_path}")
        except Exception as e:
            self.logger.error(f"✗ Failed to write all EXIF data to CSV at {self.csv_output_path}: {e}")

    def find_images(self):
        images = []
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(Path(root) / file)

        self.logger.info(f"✓ Found {len(images)} image files")
        return images

    def process_images(self):
        self.logger.info(f"Starting EXIF extraction from: {self.base_path}")

        self.connect_db()
        self.setup_csv()

        image_paths = self.find_images()

        if not image_paths:
            self.logger.warning("No images found to process.")
            return

        self.total_images = len(image_paths)
        self.start_time = time.time()

        num_workers = os.cpu_count() if os.cpu_count() else 8
        self.logger.info(f"Leveraging {num_workers} processes for parallel EXIF extraction.")

        self._display_progress()

        db_batch_temp = [] # Temporary buffer for DB inserts, gets cleared
        all_csv_data = [] # Accumulates ALL data for the final CSV write

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_image_path = {
                executor.submit(_extract_and_parse_exif_worker, str(img_path)): img_path
                for img_path in image_paths
            }

            for future in as_completed(future_to_image_path):
                original_image_path = future_to_image_path[future]
                try:
                    parsed_data = future.result()
                    if parsed_data:
                        all_csv_data.append(parsed_data) # Always add to the full CSV buffer
                        db_batch_temp.append(parsed_data) # Add to the DB batch buffer
                        self.processed_count += 1
                        self.logger.debug(f"Buffered {parsed_data['file_name']}")

                        # Check if DB batch is full and insert
                        if len(db_batch_temp) >= self.batch_size:
                            sys.stdout.write('\r' + ' ' * os.get_terminal_size().columns + '\r')
                            sys.stdout.flush()
                            self.logger.info(f"DB batch full ({len(db_batch_temp)} records). Initiating batch DB write.")
                            self._insert_batch_to_db(db_batch_temp)
                            db_batch_temp = [] # Clear DB batch buffer after insertion
                            self._display_progress()
                    else:
                        self.error_count += 1
                        self.logger.warning(f"Skipping insertion for {original_image_path.name} due to extraction/parsing errors.")

                except Exception as exc:
                    self.logger.error(f"✗ Exception caught while getting result for {original_image_path.name}: {exc}")
                    self.error_count += 1

                self._display_progress()

        # After all tasks are completed, insert any remaining data in the temporary DB buffer
        if db_batch_temp:
            sys.stdout.write('\r' + ' ' * os.get_terminal_size().columns + '\r')
            sys.stdout.flush()
            self.logger.info(f"Inserting final batch of {len(db_batch_temp)} records to DB.")
            self._insert_batch_to_db(db_batch_temp)
            db_batch_temp = [] # Clear the temporary DB buffer

        # Set the main data buffer for CSV to the accumulated data
        self._data_buffer = all_csv_data

        # Now, write all collected data (from self._data_buffer) to CSV
        sys.stdout.write('\r' + ' ' * os.get_terminal_size().columns + '\r')
        sys.stdout.flush()
        self.logger.info(f"Attempting to write all {len(self._data_buffer)} records to CSV.")
        self._write_all_to_csv()

        sys.stdout.write('\r' + ' ' * os.get_terminal_size().columns + '\r')
        sys.stdout.flush()
        self._display_progress()
        print()

        self.logger.info(f"""
        ═══════════════════════════════════════
        PROCESSING COMPLETE
        ═══════════════════════════════════════
        Total images found: {self.total_images}
        Successfully processed: {self.processed_count}
        Errors: {self.error_count}
        Success rate: {self.processed_count/self.total_images*100:.1f}%
        Database: {DB_CONFIG['database']}.exif
        CSV exported to: {self.csv_output_path if self.csv_output_path else 'Failed (check logs)'}
        ═══════════════════════════════════════
        """)

    def cleanup(self):
        if self.connection:
            self.connection.close()
        self.logger.info("Cleanup complete.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python exif2db.py /path/to/images")
        sys.exit(1)

    image_path_arg = sys.argv[1]

    if not os.path.exists(image_path_arg):
        print(f"Error: Path '{image_path_arg}' does not exist")
        sys.exit(1)
    if not os.path.isdir(image_path_arg):
        print(f"Error: Path '{image_path_arg}' is not a directory.")
        sys.exit(1)

    extractor = ExifExtractor(image_path_arg)

    try:
        extractor.process_images()
    except KeyboardInterrupt:
        sys.stdout.write('\r' + ' ' * os.get_terminal_size().columns + '\r')
        sys.stdout.flush()
        print("\n⚠ Process interrupted by user.")
        extractor.logger.warning("Process interrupted by user.")
    except Exception as e:
        sys.stdout.write('\r' + ' ' * os.get_terminal_size().columns + '\r')
        sys.stdout.flush()
        print(f"✗ Fatal error: {e}")
        extractor.logger.critical(f"Fatal error during processing: {e}", exc_info=True)
    finally:
        extractor.cleanup()

if __name__ == "__main__":
    main()
