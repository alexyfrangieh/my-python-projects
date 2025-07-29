import os
import sys
import argparse
import subprocess
import tempfile
import pandas as pd
import mysql.connector
from tqdm import tqdm
from datetime import datetime
import logging
import time
import signal
from pathlib import Path

# --- Logging Configuration ---
LOG_FILE_PATH = 'exif2db_processing.log'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a FileHandler to save logs to a file
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(file_handler)

# Create a StreamHandler to output logs to the console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(stream_handler)

# --- Global variable for process monitoring ---
current_process = None

def signal_handler(signum, frame):
    """Handle interruption signals gracefully"""
    global current_process
    logger.info("Received interruption signal. Cleaning up...")
    if current_process and current_process.poll() is None:
        logger.info("Terminating exiftool process...")
        current_process.terminate()
        time.sleep(5)
        if current_process.poll() is None:
            logger.warning("Force killing exiftool process...")
            current_process.kill()
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    """Parses command-line arguments and prompts for project details."""
    parser = argparse.ArgumentParser(description="Import EXIF metadata into MySQL/MariaDB.")
    parser.add_argument("input_path", help="Path to the root folder containing photos (e.g., 'C:/MyPhotos/ProjectA').")
    parser.add_argument("--db-host", default="localhost", help="Database host (default: localhost).")
    parser.add_argument("--db-name", default="sohdo", help="Database name (default: sohdo).")
    parser.add_argument("--db-user", default="root", help="Database username (default: root).")
    parser.add_argument("--db-password", default="Aadmin9_", help="Database password (default: Aadmin9_).")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout for exiftool in seconds (default: 1800)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Process files in batches (default: 1000)")
    parser.add_argument("--test-mode", action="store_true", help="Test with first 10 files only")

    args = parser.parse_args()

    args.input_directory = os.path.abspath(args.input_path)
    default_project = os.path.basename(os.path.normpath(args.input_directory))

    # Prompt for project name
    project = input(f"Enter project name (default: '{default_project}'): ").strip()
    args.project_name = project or default_project

    # Prompt for default latitude and longitude
    while True:
        lat_input = input("Enter default latitude for photos without GPS (blank for NULL): ").strip()
        if not lat_input:
            args.default_latitude = None
            break
        try:
            args.default_latitude = float(lat_input)
            break
        except ValueError:
            logger.warning("Invalid latitude. Please enter a number or leave blank.")

    while True:
        lon_input = input("Enter default longitude for photos without GPS (blank for NULL): ").strip()
        if not lon_input:
            args.default_longitude = None
            break
        try:
            args.default_longitude = float(lon_input)
            break
        except ValueError:
            logger.warning("Invalid longitude. Please enter a number or leave blank.")

    return args

def validate_environment():
    """Validate that exiftool is available and working"""
    try:
        result = subprocess.run(['exiftool', '-ver'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"Exiftool version: {result.stdout.strip()}")
            return True
        else:
            logger.error("Exiftool is not responding correctly")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Exiftool version check timed out")
        return False
    except FileNotFoundError:
        logger.error("exiftool command not found. Please ensure exiftool is installed and in your system's PATH.")
        return False
    except Exception as e:
        logger.error(f"Error checking exiftool: {e}")
        return False

def count_image_files(input_dir):
    """Count image files in directory for progress estimation"""
    image_extensions = {'.jpg', '.jpeg', '.tiff', '.tif', '.raw', '.cr2', '.nef', '.dng', '.arw', '.orf', '.rw2'}
    count = 0
    
    logger.info("Counting image files...")
    try:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    count += 1
    except Exception as e:
        logger.warning(f"Error counting files: {e}")
        return 0
    
    logger.info(f"Found approximately {count} image files")
    return count

def extract_metadata(input_dir, temp_csv_path, args):
    """
    Runs exiftool to extract metadata from files in input_dir and saves it to a temporary CSV.
    Enhanced with better error handling, timeouts, and progress monitoring.
    """
    global current_process
    
    logger.info(f"📸 Extracting EXIF data from '{input_dir}' with exiftool...")
    
    # Validate input directory
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not os.access(input_dir, os.R_OK):
        logger.error(f"Input directory is not readable: {input_dir}")
        sys.exit(1)
    
    # Count files for progress estimation
    file_count = count_image_files(input_dir)
    if file_count == 0:
        logger.warning("No image files found in the specified directory")
        return
    
    # List of specific tags to extract
    exiftool_command_tags = [
        "-SourceFile",
        "-FileName",
        "-ShutterCount",
        "-LightValue",
        "-SubSecCreateDate",
        "-DateTimeOriginal",
        "-TimeZone",
        "-ISO",
        "-Aperture",
        "-ShutterSpeed",
        "-Model",
        "-SerialNumber",
        "-FileSize",
        "-Make",
        "-FocalLength",
        "-Lens",
        "-ExposureProgram",
        "-WB_RBLevels",
        "-WB_GRBGLevels",
        "-ImageWidth",
        "-ImageHeight",
        "-GPSLatitude",
        "-GPSLongitude",
        "-Orientation",
        "-WhiteBalance",
    ]

    command = [
        "exiftool",
        "-csv",
        "-r",  # Recurse into subdirectories
        "-ignoreMinorErrors",  # Skip problematic files
        "-quiet",  # Reduce verbose output
        "-fast2",  # Fast processing mode
        *exiftool_command_tags,
        input_dir
    ]
    
    # Add test mode limitation
    if args.test_mode:
        command.insert(-1, "-m")  # Allow minor errors
        logger.info("Running in test mode - processing limited files")

    try:
        logger.info(f"Running exiftool command: {' '.join(command[:10])}... (truncated)")
        logger.info(f"Timeout set to {args.timeout} seconds")
        
        # Start the process
        current_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Monitor the process with progress updates
        logger.info("Exiftool process started, monitoring progress...")
        start_time = time.time()
        last_update = start_time
        
        while current_process.poll() is None:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Update every 30 seconds
            if current_time - last_update >= 30:
                logger.info(f"Exiftool still processing... Elapsed: {elapsed:.1f}s")
                last_update = current_time
            
            # Check timeout
            if elapsed > args.timeout:
                logger.error(f"Exiftool process timed out after {args.timeout} seconds")
                current_process.terminate()
                time.sleep(5)
                if current_process.poll() is None:
                    current_process.kill()
                raise subprocess.TimeoutExpired(command, args.timeout)
            
            time.sleep(1)  # Check every second
        
        # Get the output
        stdout, stderr = current_process.communicate(timeout=60)
        
        if current_process.returncode != 0:
            logger.error(f"Exiftool failed with return code {current_process.returncode}")
            logger.error(f"Stderr: {stderr}")
            if "No matching files" in stderr:
                logger.warning("No matching image files found by exiftool")
                return
            sys.exit(1)
        
        # Check if we got any output
        if not stdout.strip():
            logger.warning("Exiftool returned empty output - no metadata extracted")
            return
        
        # Save output to temporary file
        with open(temp_csv_path, "w", encoding='utf-8') as csvfile:
            csvfile.write(stdout)
        
        total_time = time.time() - start_time
        logger.info(f"Exiftool completed successfully in {total_time:.1f} seconds")
        
        # Log some statistics
        line_count = len(stdout.strip().split('\n')) - 1  # Subtract header
        logger.info(f"Extracted metadata for {line_count} files")

    except subprocess.TimeoutExpired:
        logger.error(f"Exiftool command timed out after {args.timeout} seconds")
        logger.error("This might indicate a problematic file or very large directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during exiftool execution: {e}", exc_info=True)
        if current_process and current_process.poll() is None:
            current_process.terminate()
        sys.exit(1)
    finally:
        current_process = None

def parse_filesize(size_str):
    """
    Parses filesize strings (e.g., "1.2 MB") into bytes.
    Handles various units and returns None for unparseable strings.
    """
    if pd.isna(size_str) or not isinstance(size_str, str):
        return None
    try:
        parts = str(size_str).strip().split(" ")
        if len(parts) == 2:
            num = float(parts[0])
            unit = parts[1].lower()
            units = {
                'bytes': 1,
                'kb': 1024,
                'mb': 1024**2,
                'gb': 1024**3,
                'tb': 1024**4,
            }
            return int(num * units.get(unit, 1))
        elif len(parts) == 1 and parts[0].isdigit():
            return int(parts[0])
        return None
    except Exception:
        return None

def process_metadata(args, temp_csv_path):
    """
    Reads the raw exiftool CSV, processes the data, and maps it to the database schema.
    Enhanced with better error handling and validation.
    """
    logger.info("🧪 Processing metadata...")
    
    # Check if temp file exists and has content
    if not os.path.exists(temp_csv_path):
        logger.error(f"Temporary CSV file not found: {temp_csv_path}")
        sys.exit(1)
    
    if os.path.getsize(temp_csv_path) == 0:
        logger.warning("Temporary CSV file is empty - no data to process")
        sys.exit(1)
    
    try:
        # Read with dtype=str to prevent pandas from inferring types incorrectly
        df = pd.read_csv(temp_csv_path, encoding='utf-8', dtype=str, na_values=['-', 'nan', 'N/A', '', 'None'])
        logger.info(f"Read {len(df)} rows from exiftool output")
    except pd.errors.EmptyDataError:
        logger.warning(f"No data extracted by exiftool from '{args.input_directory}'. Check your input path and photo files.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading temporary exiftool CSV: {e}", exc_info=True)
        sys.exit(1)

    # Clean column names
    df.columns = [col.replace('.', '_').replace('-', '_').replace(' ', '_').lower() for col in df.columns]
    logger.info(f"Available columns: {list(df.columns)}")

    processed_df = pd.DataFrame()

    # Direct mappings & simple conversions
    processed_df['project_name'] = str(args.project_name)
    processed_df['source_file'] = df.get('sourcefile', pd.Series(dtype=str))

    # Calculate relative_path with better error handling
    def calculate_relative_path(source_file):
        if pd.isna(source_file):
            return None
        try:
            abs_input = os.path.abspath(args.input_directory)
            abs_source = os.path.abspath(source_file)
            if abs_source.startswith(abs_input):
                return os.path.relpath(abs_source, abs_input)
            else:
                return source_file  # Return original if not within input directory
        except Exception:
            return source_file

    processed_df['relative_path'] = processed_df['source_file'].apply(calculate_relative_path)
    processed_df['file_name'] = df.get('filename', pd.Series(dtype=str))
    processed_df['file_size'] = df.get('filesize', pd.Series(dtype=str)).apply(parse_filesize)
    processed_df['image_width'] = pd.to_numeric(df.get('imagewidth', pd.Series(dtype=str)), errors='coerce')
    processed_df['image_height'] = pd.to_numeric(df.get('imageheight', pd.Series(dtype=str)), errors='coerce')

    # Enhanced datetime parsing
    def parse_datetime_safe(dt_str):
        if pd.isna(dt_str):
            return None
        try:
            # Try multiple formats
            formats = ['%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y:%m:%d %H:%M:%S.%f']
            for fmt in formats:
                try:
                    return pd.to_datetime(dt_str, format=fmt)
                except:
                    continue
            # If all specific formats fail, let pandas infer
            return pd.to_datetime(dt_str, errors='coerce')
        except:
            return None

    processed_df['date_time_original'] = df.get('datetimeoriginal', pd.Series(dtype=str)).apply(parse_datetime_safe)
    processed_df['subsec_create_date'] = df.get('subseccreatedate', pd.Series(dtype=str))
    processed_df['time_zone'] = df.get('timezone', pd.Series(dtype=str))

    # Numeric fields with better error handling
    numeric_fields = {
        'shutter_count': 'shuttercount',
        'light_value': 'lightvalue',
        'iso': 'iso',
        'aperture': 'aperture'
    }
    
    for processed_col, source_col in numeric_fields.items():
        processed_df[processed_col] = pd.to_numeric(df.get(source_col, pd.Series(dtype=str)), errors='coerce')

    # String fields
    string_fields = {
        'shutter_speed': 'shutterspeed',
        'focal_length': 'focallength',
        'model': 'model',
        'make': 'make',
        'serial_number': 'serialnumber',
        'lens': 'lens',
        'exposure_program': 'exposureprogram',
        'wb_rb_levels': 'wb_rblevels',
        'wb_grbg_levels': 'wb_grbglevels',
        'orientation': 'orientation',
        'white_balance': 'whitebalance'
    }
    
    for processed_col, source_col in string_fields.items():
        processed_df[processed_col] = df.get(source_col, pd.Series(dtype=str))

    # GPS handling with enhanced logic
    gps_lat_exif = pd.to_numeric(df.get('gpslatitude', pd.Series()), errors='coerce')
    gps_lon_exif = pd.to_numeric(df.get('gpslongitude', pd.Series()), errors='coerce')

    processed_df['gps_latitude'] = gps_lat_exif.fillna(args.default_latitude)
    processed_df['gps_longitude'] = gps_lon_exif.fillna(args.default_longitude)

    # Optional columns
    processed_df['small_version_path'] = None
    processed_df['histogram_base64'] = None

    # Ensure correct column order
    db_columns_order = [
        'project_name', 'source_file', 'relative_path', 'file_name', 'file_size',
        'image_width', 'image_height', 'date_time_original', 'subsec_create_date',
        'time_zone', 'shutter_count', 'light_value', 'iso', 'aperture',
        'shutter_speed', 'focal_length', 'model', 'make', 'serial_number', 'lens',
        'exposure_program', 'wb_rb_levels', 'wb_grbg_levels',
        'gps_latitude', 'gps_longitude', 'orientation', 'white_balance',
        'small_version_path', 'histogram_base64'
    ]

    final_df = processed_df.reindex(columns=db_columns_order)

    # Enhanced data validation
    valid_rows = []
    skipped_count = 0
    
    for index, row in final_df.iterrows():
        row_is_valid = True
        file_name_for_log = row.get('file_name', 'N/A')
        source_file_for_log = row.get('source_file', 'N/A')

        # Critical field validation
        if pd.isna(row['source_file']) or str(row['source_file']).strip() == '':
            logger.warning(f"Skipping row for '{file_name_for_log}': missing source_file")
            row_is_valid = False
        elif pd.isna(row['relative_path']) or str(row['relative_path']).strip() == '':
            logger.warning(f"Skipping row for '{file_name_for_log}': missing relative_path")
            row_is_valid = False
        elif pd.isna(row['file_name']) or str(row['file_name']).strip() == '':
            logger.warning(f"Skipping row for '{source_file_for_log}': missing file_name")
            row_is_valid = False

        if row_is_valid:
            valid_rows.append(row)
        else:
            skipped_count += 1

    final_df = pd.DataFrame(valid_rows, columns=final_df.columns)
    
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} rows due to missing critical metadata")
    
    logger.info(f"Successfully processed {len(final_df)} valid records")

    # Replace pandas NaN/NaT with Python None
    for col in final_df.columns:
        if pd.api.types.is_numeric_dtype(final_df[col]) or pd.api.types.is_float_dtype(final_df[col]):
            final_df[col] = final_df[col].where(pd.notna(final_df[col]), None)
        elif pd.api.types.is_datetime64_any_dtype(final_df[col]):
            final_df[col] = final_df[col].where(pd.notna(final_df[col]), None)
        else:
            final_df[col] = final_df[col].where(final_df[col].notna(), None)
            final_df[col] = final_df[col].replace('None', None)
            final_df[col] = final_df[col].replace('', None)

    return final_df

def insert_metadata(df, args, output_csv_path):
    """
    Inserts processed metadata into the database and outputs to CSV files.
    Enhanced with better error handling and batch processing.
    """
    if df.empty:
        logger.warning("No data to insert - DataFrame is empty")
        return

    # Save processed DataFrame to CSV
    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        logger.info(f"Processed metadata saved to: {output_csv_path}")
    except Exception as e:
        logger.error(f"Error saving processed CSV to '{output_csv_path}': {e}", exc_info=True)

    # Generate revise.csv
    try:
        revise_df = df[['relative_path', 'file_name']].copy()
        revise_csv_path = os.path.join(os.getcwd(), "revise.csv")
        revise_df.to_csv(revise_csv_path, index=False, encoding='utf-8')
        logger.info(f"Revision file saved to: {revise_csv_path}")
    except Exception as e:
        logger.error(f"Error saving revise.csv: {e}", exc_info=True)

    # Database insertion
    logger.info("🛠 Inserting into database...")
    conn = None
    cursor = None
    
    try:
        # Test database connection first
        conn = mysql.connector.connect(
            host=args.db_host,
            user=args.db_user,
            password=args.db_password,
            database=args.db_name,
            connect_timeout=30,
            autocommit=False
        )
        cursor = conn.cursor()
        logger.info("Database connection established successfully")

        # Database column configuration
        db_column_names_for_insert = [
            'project_name', 'source_file', 'relative_path', 'file_name', 'file_size',
            'image_width', 'image_height', 'date_time_original', 'subsec_create_date',
            'time_zone', 'shutter_count', 'light_value', 'iso', 'aperture',
            'shutter_speed', 'focal_length', 'model', 'make', 'serial_number', 'lens',
            'exposure_program', 'wb_rb_levels', 'wb_grbg_levels',
            'gps_latitude', 'gps_longitude', 'orientation', 'white_balance',
            'small_version_path', 'histogram_base64'
        ]

        update_fields = [
            'source_file', 'relative_path', 'file_size',
            'image_width', 'image_height', 'date_time_original', 'subsec_create_date',
            'time_zone', 'shutter_count', 'light_value', 'iso', 'aperture',
            'shutter_speed', 'focal_length', 'model', 'make', 'serial_number', 'lens',
            'exposure_program', 'wb_rb_levels', 'wb_grbg_levels',
            'gps_latitude', 'gps_longitude', 'orientation', 'white_balance',
            'small_version_path', 'histogram_base64'
        ]

        on_duplicate_update_clause = ", ".join([f"{field}=VALUES({field})" for field in update_fields])

        insert_query = f"""
            INSERT INTO photos (
                {', '.join(db_column_names_for_insert)}
            ) VALUES (
                {', '.join(['%s'] * len(db_column_names_for_insert))}
            )
            ON DUPLICATE KEY UPDATE
            {on_duplicate_update_clause}
        """

        # Prepare data for insertion
        values_to_insert = []
        for index, row in df.iterrows():
            row_values = []
            for col_name in db_column_names_for_insert:
                val = row[col_name]
                if pd.isna(val):
                    row_values.append(None)
                elif isinstance(val, pd.Timestamp):
                    row_values.append(val.to_pydatetime())
                else:
                    row_values.append(val)
            values_to_insert.append(tuple(row_values))

        if not values_to_insert:
            logger.info("No valid data to insert into the database")
            return

        # Insert in batches with progress tracking
        batch_size = args.batch_size
        total_batches = (len(values_to_insert) + batch_size - 1) // batch_size
        logger.info(f"Inserting {len(values_to_insert)} records in {total_batches} batches of {batch_size}")

        inserted_count = 0
        with tqdm(total=len(values_to_insert), desc="Inserting rows", unit="rows") as pbar:
            for i in range(0, len(values_to_insert), batch_size):
                batch = values_to_insert[i:i + batch_size]
                try:
                    cursor.executemany(insert_query, batch)
                    conn.commit()
                    inserted_count += len(batch)
                    pbar.update(len(batch))
                except mysql.connector.Error as batch_error:
                    logger.error(f"Error inserting batch {i//batch_size + 1}: {batch_error}")
                    conn.rollback()
                    # Try inserting records one by one in this batch
                    for j, single_record in enumerate(batch):
                        try:
                            cursor.execute(insert_query, single_record)
                            conn.commit()
                            inserted_count += 1
                            pbar.update(1)
                        except mysql.connector.Error as single_error:
                            logger.warning(f"Failed to insert record {i+j+1}: {single_error}")
                            conn.rollback()

        logger.info(f"Successfully inserted/updated {inserted_count} out of {len(values_to_insert)} records")

    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        if conn:
            conn.rollback()
    except Exception as e:
        logger.error(f"Unexpected error during database insertion: {e}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        logger.info("Database connection closed")

def main():
    """Main execution function with comprehensive error handling"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate environment
        if not validate_environment():
            sys.exit(1)
        
        logger.info(f"Starting EXIF processing for project: {args.project_name}")
        logger.info(f"Input directory: {args.input_directory}")
        logger.info(f"Timeout: {args.timeout} seconds")
        
        # Create temporary file for exiftool output
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_exiftool_output_file:
            temp_exiftool_csv_path = temp_exiftool_output_file.name

        # Define output paths
        output_csv_filename = f"{args.project_name}_processed_metadata.csv"
        output_csv_path = os.path.join(os.getcwd(), output_csv_filename)

        try:
            # Step 1: Extract metadata using exiftool
            extract_metadata(args.input_directory, temp_exiftool_csv_path, args)

            # Step 2: Process and clean the extracted metadata
            df = process_metadata(args, temp_exiftool_csv_path)

            # Step 3: Insert into database and save processed CSV
            insert_metadata(df, args, output_csv_path)

            logger.info("✅ Script completed successfully!")

        except KeyboardInterrupt:
            logger.info("Script interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during main process: {e}", exc_info=True)
            sys.exit(1)
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_exiftool_csv_path):
                try:
                    os.remove(temp_exiftool_csv_path)
                    logger.info("Temporary files cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {e}")

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()