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

class ExifExtractor:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.connection = None
        self.csv_file = None
        self.csv_writer = None
        self.processed_count = 0
        self.error_count = 0

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('exif_extraction.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def connect_db(self):
        """Connect to MySQL database"""
        try:
            self.connection = pymysql.connect(**DB_CONFIG)
            self.logger.info("✓ Connected to MySQL database")
            self.create_table()
        except Exception as e:
            self.logger.error(f"✗ Database connection failed: {e}")
            sys.exit(1)

    def create_table(self):
        """Create comprehensive EXIF table based on actual Nikon D3300 output"""
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
                cursor.execute("DROP TABLE IF EXISTS exif")  # Fresh start
                cursor.execute(create_table_sql)
                self.connection.commit()
                self.logger.info("✓ Fresh EXIF table created successfully")
        except Exception as e:
            self.logger.error(f"✗ Table creation failed: {e}")
            sys.exit(1)

    def setup_csv(self):
        """Setup CSV file for export"""
        try:
            csv_path = self.base_path / 'exif_data.csv'
            self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            self.logger.info(f"✓ CSV file created: {csv_path}")
        except PermissionError:
            csv_path = Path.cwd() / 'exif_data.csv'
            try:
                self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
                self.logger.info(f"✓ CSV file created (fallback): {csv_path}")
            except Exception as e:
                self.logger.error(f"✗ CSV setup failed even with fallback: {e}")
                return
        except Exception as e:
            self.logger.error(f"✗ CSV setup failed: {e}")
            return

        # CSV fieldnames matching database columns
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

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()

    def extract_exif(self, image_path):
        """Extract EXIF data using exiftool"""
        try:
            cmd = ['exiftool', '-json', '-charset', 'utf8', str(image_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                self.logger.warning(f"⚠ exiftool failed for {image_path}: {result.stderr}")
                return None

            exif_data = json.loads(result.stdout)[0]
            self.logger.debug(f"✓ Extracted EXIF from {image_path.name}")
            return exif_data

        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ Timeout extracting EXIF from {image_path}")
            return None
        except Exception as e:
            self.logger.error(f"✗ EXIF extraction failed for {image_path}: {e}")
            return None

    def safe_convert_int(self, value):
        """Safely convert value to int, return None if fails"""
        if value is None:
            return None
        try:
            # Handle string numbers and remove any non-numeric suffixes
            str_val = str(value).strip()
            # Extract just the number part (useful for values like "6000x4000")
            import re
            numbers = re.findall(r'\d+', str_val)
            if numbers:
                return int(numbers[0])
            return None
        except:
            return None

    def parse_gps_coord(self, coord_str):
        """Parse GPS coordinates"""
        if not coord_str:
            return None
        try:
            if isinstance(coord_str, (int, float)):
                return float(coord_str)
            # Handle various GPS formats
            coord_str = str(coord_str)
            if 'deg' in coord_str or "'" in coord_str:
                import re
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

    def parse_exif_data(self, exif_raw, image_path):
        """Parse EXIF data into structured format - flexible text approach"""
        def safe_get(key, default=None):
            return exif_raw.get(key, default)

        # Parse datetime with multiple format support
        def parse_datetime(date_str):
            if not date_str:
                return None
            try:
                date_str = str(date_str)
                # Remove timezone info for MySQL compatibility
                if '+' in date_str:
                    date_str = date_str.split('+')[0]
                if '-' in date_str and len(date_str) > 19:
                    date_str = date_str[:19]

                for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        continue
                return date_str  # Return as-is if can't parse
            except:
                return str(date_str) if date_str else None

        parsed_data = {
            # File Information
            'source_file': str(image_path),
            'file_name': image_path.name,
            'directory': str(safe_get('Directory', '')),
            'file_size': str(safe_get('FileSize', '')),
            'file_modify_date': str(safe_get('FileModifyDate', '')),
            'file_type': str(safe_get('FileType', '')),
            'mime_type': str(safe_get('MIMEType', '')),

            # Camera Information
            'make': str(safe_get('Make', '')),
            'model': str(safe_get('Model', '')),
            'serial_number': str(safe_get('SerialNumber', '')),
            'firmware_version': str(safe_get('FirmwareVersion', '')),
            'software': str(safe_get('Software', '')),

            # Image Settings
            'image_width': self.safe_convert_int(safe_get('ImageWidth') or safe_get('ExifImageWidth')),
            'image_height': self.safe_convert_int(safe_get('ImageHeight') or safe_get('ExifImageHeight')),
            'orientation': str(safe_get('Orientation', '')),
            'resolution_unit': str(safe_get('ResolutionUnit', '')),
            'x_resolution': str(safe_get('XResolution', '')),
            'y_resolution': str(safe_get('YResolution', '')),
            'color_space': str(safe_get('ColorSpace', '')),

            # Exposure Settings
            'exposure_time': str(safe_get('ExposureTime', '')),
            'f_number': str(safe_get('FNumber', '')),
            'aperture': str(safe_get('Aperture', '')),
            'iso_speed': str(safe_get('ISO', '')),
            'iso_setting': str(safe_get('ISOSetting', '')),
            'exposure_program': str(safe_get('ExposureProgram', '')),
            'exposure_mode': str(safe_get('ExposureMode', '')),
            'exposure_compensation': str(safe_get('ExposureCompensation', '')),
            'metering_mode': str(safe_get('MeteringMode', '')),

            # Date/Time
            'date_time_original': parse_datetime(safe_get('DateTimeOriginal')),
            'create_date': parse_datetime(safe_get('CreateDate')),
            'modify_date': parse_datetime(safe_get('ModifyDate')),
            'sub_sec_time': str(safe_get('SubSecTime', '')),
            'sub_sec_create_date': str(safe_get('SubSecCreateDate', '')),

            # Flash Settings
            'flash': str(safe_get('Flash', '')),
            'flash_mode': str(safe_get('FlashMode', '')),
            'flash_setting': str(safe_get('FlashSetting', '')),
            'flash_type': str(safe_get('FlashType', '')),
            'flash_compensation': str(safe_get('FlashCompensation', '')),
            'flash_exposure_comp': str(safe_get('FlashExposureComp', '')),

            # Focus Settings
            'focus_mode': str(safe_get('FocusMode', '')),
            'af_area_mode': str(safe_get('AFAreaMode', '')),
            'focus_distance': str(safe_get('FocusDistance', '')),
            'focus_position': str(safe_get('FocusPosition', '')),

            # Lens Information
            'lens': str(safe_get('Lens', '')),
            'lens_type': str(safe_get('LensType', '')),
            'lens_id': str(safe_get('LensID', '')),
            'focal_length': str(safe_get('FocalLength', '')),
            'focal_length_35mm': str(safe_get('FocalLengthIn35mmFormat', '')),
            'min_focal_length': str(safe_get('MinFocalLength', '')),
            'max_focal_length': str(safe_get('MaxFocalLength', '')),
            'max_aperture_value': str(safe_get('MaxApertureValue', '')),

            # White Balance
            'white_balance': str(safe_get('WhiteBalance', '')),
            'white_balance_fine_tune': str(safe_get('WhiteBalanceFineTune', '')),
            'wb_rb_levels': str(safe_get('WB_RBLevels', '')),
            'wb_rggb_levels': str(safe_get('WB_RGGBLevels', '')),

            # Picture Settings
            'quality': str(safe_get('Quality', '')),
            'shooting_mode': str(safe_get('ShootingMode', '')),
            'picture_control_name': str(safe_get('PictureControlName', '')),
            'picture_control_base': str(safe_get('PictureControlBase', '')),
            'contrast': str(safe_get('Contrast', '')),
            'saturation': str(safe_get('Saturation', '')),
            'sharpness': str(safe_get('Sharpness', '')),
            'brightness': str(safe_get('Brightness', '')),

            # GPS Data
            'gps_latitude': self.parse_gps_coord(safe_get('GPSLatitude')),
            'gps_longitude': self.parse_gps_coord(safe_get('GPSLongitude')),
            'gps_altitude': self.parse_gps_coord(safe_get('GPSAltitude')),

            # Technical Details
            'shutter_count': self.safe_convert_int(safe_get('ShutterCount')),
            'light_value': str(safe_get('LightValue', '')),
            'noise_reduction': str(safe_get('NoiseReduction', '')),
            'vibration_reduction': str(safe_get('VibrationReduction', '')),
            'active_d_lighting': str(safe_get('ActiveD-Lighting', '')),
            'scene_capture_type': str(safe_get('SceneCaptureType', '')),

            # Raw data and processing
            'exif_raw': json.dumps(exif_raw, indent=2),
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return parsed_data

    def insert_to_db(self, data):
        """Insert parsed data into database"""
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
                cursor.execute(insert_sql, data)
                self.connection.commit()
                self.logger.info(f"✓ Inserted to DB: {data['file_name']} | Camera: {data['make']} {data['model']} | ISO: {data['iso_speed']} | f/{data['f_number']} | {data['exposure_time']}s")
                return True
        except Exception as e:
            self.logger.error(f"✗ DB insertion failed for {data['file_name']}: {e}")
            return False

    def write_to_csv(self, data):
        """Write data to CSV file"""
        if not self.csv_writer:
            return

        try:
            # Remove JSON field for CSV and convert None to empty string
            csv_data = {k: (v if v is not None else '') for k, v in data.items() if k != 'exif_raw'}
            self.csv_writer.writerow(csv_data)
            self.csv_file.flush()
            self.logger.debug(f"✓ Written to CSV: {data['file_name']}")
        except Exception as e:
            self.logger.error(f"✗ CSV write failed for {data['file_name']}: {e}")

    def find_images(self):
        """Find all image files in directory and subdirectories"""
        images = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(Path(root) / file)

        self.logger.info(f"✓ Found {len(images)} image files")
        return images

    def process_images(self):
        """Main processing function"""
        self.logger.info(f"Starting EXIF extraction from: {self.base_path}")

        # Setup connections
        self.connect_db()
        self.setup_csv()

        # Find images
        images = self.find_images()

        if not images:
            self.logger.warning("No images found!")
            return

        # Process each image
        for i, image_path in enumerate(images, 1):
            self.logger.info(f"Processing {i}/{len(images)}: {image_path.name}")

            try:
                # Extract EXIF
                exif_raw = self.extract_exif(image_path)
                if not exif_raw:
                    self.error_count += 1
                    continue

                # Parse data
                parsed_data = self.parse_exif_data(exif_raw, image_path)

                # Insert to database
                if self.insert_to_db(parsed_data):
                    self.processed_count += 1

                # Write to CSV
                self.write_to_csv(parsed_data)

                # Progress update
                if i % 5 == 0:
                    self.logger.info(f"Progress: {i}/{len(images)} ({i/len(images)*100:.1f}%)")

            except Exception as e:
                self.logger.error(f"✗ Failed processing {image_path}: {e}")
                self.error_count += 1

        # Summary
        self.logger.info(f"""
        ═══════════════════════════════════════
        PROCESSING COMPLETE
        ═══════════════════════════════════════
        Total images found: {len(images)}
        Successfully processed: {self.processed_count}
        Errors: {self.error_count}
        Success rate: {self.processed_count/len(images)*100:.1f}%
        Database: {DB_CONFIG['database']}.exif
        CSV exported: {self.csv_file.name if self.csv_file else 'Failed'}
        ═══════════════════════════════════════
        """)

    def cleanup(self):
        """Cleanup connections"""
        if self.csv_file:
            self.csv_file.close()
        if self.connection:
            self.connection.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python exif2db.py /path/to/images")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Path '{image_path}' does not exist")
        sys.exit(1)

    extractor = ExifExtractor(image_path)

    try:
        extractor.process_images()
    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
    except Exception as e:
        print(f"✗ Fatal error: {e}")
    finally:
        extractor.cleanup()

if __name__ == "__main__":
    main()
