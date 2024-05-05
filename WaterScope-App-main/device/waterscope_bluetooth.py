import logging
import argparse
import csv
import time
import random

import json
import hashlib

import pydbus
import socket
import traceback

bus = pydbus.SystemBus()

sender_id = 0

package_size = 0


def package_offsets(size, package_size):
    return range(0, size, package_size)


def package_end(size, offset, package_size):
    end = offset + package_size
    return end if end < size else size


# noinspection PyShadowingBuiltins
def info_matches(response, id, type, part):
    try:
        return response['id'] == id and response['type'] == type and response['part'] == part
    except KeyError:
        return False


def calculate_progress(size, offset):
    return round((offset / size) * 100, 2)


# noinspection PyShadowingBuiltins
def send_file_response(connection, instruction, file_path, package_size, type, id=None):
    logging.info(f'Sending file "{file_path}"')

    with open(file_path, 'rb') as file:
        data = file.read()

    size = len(data)
    hashes = []

    if package_size < 1:
        package_size = size

    for offset in package_offsets(size, package_size):
        hashes.append(hashlib.sha1(data[offset:package_end(size, offset, package_size)]).hexdigest())

    logging.debug(f'{len(hashes)} file hash{"es" if len(hashes) > 1 else ""} created for "{file_path}".')

    info = {
        'id': id,
        'type': type,
        'hashes': hashes,
        'packageSize': package_size,
        'size': len(data)
    }

    send_response(connection, instruction, payload=json.dumps(info).encode('utf-8'))

    with open(file_path, 'rb') as file:
        for offset in package_offsets(size, package_size):
            logging.debug(
                f'Sending file "{file_path}" from offset {offset}, '
                f'which is equal to {calculate_progress(size, offset)}%.'
            )

            try:
                connection.sendfile(file, offset, package_end(size, offset, package_size) - offset)

                status = None
                payload = {'part': None}
                part = offset // package_size
                connection.settimeout(30)

                while not (status == 'ok' and info_matches(payload, id, type, part)):
                    response = json.loads(connection.recv(1024).decode('utf-8'))

                    if "instruction" in response:
                        logging.error(
                            f'Received new instruction before file "{file_path}" was fully submitted. '
                            f'Skipping instruction "{response["instruction"]}" and aborting transmission.'
                        )
                        return

                    payload = read_payload(connection, response)

                    if payload is None:
                        logging.error('Could not read file part response payload. Aborting file transmission.')
                        return

                    payload = json.loads(payload.decode('utf-8'))

                    if info_matches(payload, id, type, part):
                        try:
                            status = response['status']
                        except KeyError:
                            continue

                        if status == 'invalid':
                            logging.warning(
                                f'File "{file_path}" got corrupted starting at offset {offset}, '
                                f'which is equal to {calculate_progress(size, offset)}%. '
                                f'Resending file starting from this offset.'
                            )
                            connection.sendfile(file, offset, package_end(size, offset, package_size) - offset)
                        elif status == 'failed':
                            logging.error(
                                f'Sending file "{file_path}" failed '
                                f'after {calculate_progress(size, offset)}% has been send.'
                            )
                            return
            except socket.timeout:
                logging.error(
                    f'Sending file "{file_path}" timed out after {calculate_progress(size, offset)}% has been send.'
                )
                return
            except OSError:
                raise ConnectionError
            finally:
                connection.settimeout(None)

    logging.info(f'Successful send file "{file_path}".')


def send_sample_update(connection, sample_id, sample_status, result=None):
    logging.info(f'Sample #{sample_id} reached status "{sample_status}".')

    payload = json.dumps({
        'id': sample_id,
        'status': sample_status,
        'result': result,
    }).encode('utf-8')

    send_instruction(connection, 'sample', payload=payload)


# noinspection PyShadowingBuiltins
def send_sample(connection, instruction, id):
    logging.info(f'Sample #{id} requested.')

    global samples

    try:
        send_response(connection, instruction, payload=json.dumps(samples[id]).encode('utf-8'))
    except KeyError:
        send_response(connection, instruction, status=f'Sample #{id} not found')


# noinspection PyShadowingBuiltins
def analyse_sample(connection, instruction, id, data):
    logging.info(f'''Sample #{id} submitted.\n{json.dumps({
        'location': data['location'],
        'time': data['time'],
        'comment': data['comment'],
        'coordinates': data['coordinates']
    }, indent=4)}''')

    send_response(connection, instruction)
    time.sleep(1)

    try:
        send_sample_update(connection, id, 'analysing')
        time.sleep(1)
        send_sample_update(connection, id, 'defogging')
        time.sleep(1)
        send_sample_update(connection, id, 'autofocusing')
        time.sleep(1)
        send_sample_update(connection, id, 'image capturing')
        time.sleep(1)
        send_sample_update(connection, id, 'counting')
        time.sleep(1)

        result = {
            'eColiform': random.randint(0, 150),
            'otherColiform': random.randint(0, 150),
            'flag': random.choice(['normal', 'too_many', 'overgrown', 'new', 'unsure', 'anomalous']),
        }

        send_sample_update(connection, id, 'result', result)

    except TimeoutError:
        return


def sample(connection, instruction):
    data = read_payload(connection, instruction)

    if data is None:
        logging.error('Could not read sample instruction payload.')
        return

    data = json.loads(data)

    try:
        action = data['action']
        # noinspection PyShadowingBuiltins
        id = data['id']
    except KeyError:
        send_response(connection, instruction, status=f'A sample action is missing')
        return

    global args

    if action == 'submit':
        analyse_sample(connection, instruction, id, data)
    elif action == 'get':
        send_sample(connection, instruction, id)
    elif action == 'get raw image':
        try:
            send_file_response(
                connection,
                instruction,
                args.raw,
                args.package_size * 1024,
                'raw image',
                id=id,
            )
        except TimeoutError:
            pass
    elif action == 'get preview image':
        try:
            send_file_response(
                connection,
                instruction,
                args.preview,
                args.package_size * 1024,
                'preview image',
                id=id,
            )
        except TimeoutError:
            pass
    else:
        send_response(connection, instruction, status=f'Sample action "{action}" not supported')


def diagnostics():
    return {
        'temperature': 36.5,
        'servo': 'OK',
        'defogger': 'OK',
        'incubator': 'Too hot alert 02/03/20',
        'deviceTestCount': 50,
        'batteryLevel': 69,
        'averageFocusingPosition': 150.0,
        'softwareVersion': '3.4',
        'firmwareVersion': '4.2',
        'location': 'Tanzania',
        'internet': 'IoT/No Wi',
    }


def sample_history(sample_file_path):
    global args

    entries = {}

    try:
        with open(sample_file_path, newline='') as file:
            reader = csv.reader(file)
            next(reader, None)

            i = 0

            for row in reader:
                i += 1

                try:
                    hours = row[10].split(':')[0]
                    minutes = row[10].split(':')[1].split(' ')[0]

                    entries[int(row[0])] = {
                        'id': int(row[0]),
                        'eColiform': int(row[1]),
                        'otherColiform': int(row[2]),
                        'location': row[8],
                        'time': int(hours) * 60 + int(minutes),
                        'comment': row[11],
                        'flag': row[3],
                    }
                except (IndexError, ValueError):
                    logging.error(f'Could not parse sample in row #{i + 1}')
                    continue
    except FileNotFoundError:
        pass

    logging.debug(f'Loaded samples from csv-file\n{json.dumps(entries, indent=4)}')

    return entries


def calculate_payload_validation(payload):
    return {
        'size': len(payload) if payload is not None else 0,
        'checksum': hashlib.sha1(payload).hexdigest() if payload is not None else None
    }


def send_instruction(connection, instruction_name, payload=None):
    global sender_id
    sender_id += 1

    # noinspection PyShadowingBuiltins
    id = 'w' + str(sender_id)  # adding an character so they don't clash with the IDs send by the app

    instruction = {
        'id': id,
        'instruction': instruction_name,
        'payload': calculate_payload_validation(payload)
    }

    logging.debug(f'Sending instruction.\n{json.dumps(instruction, indent=4)}')

    connection.sendall(json.dumps(instruction).encode("utf-8"))

    connection.settimeout(30)

    if payload is not None:
        connection.sendall(payload)

    try:
        response = {'status': None, 'id': None}

        while not (response['status'] == 'ok' and response['id'] == id):
            try:
                response = json.loads(connection.recv(1024).decode('utf-8'))

                if response['id'] == id:
                    if response['status'] == 'invalid':
                        logging.warning(f'Retrying instruction "{instruction_name}" with ID "{id}".')
                        send_instruction(connection, instruction_name, payload=payload)
                    elif response['status'] != 'ok':
                        logging.error(
                            f'Instruction "{instruction_name}" with ID "{id}" '
                            f'failed with status {response["status"]}.'
                        )
                        break

            except socket.timeout:
                logging.error(f'Instruction "{instruction_name}" with ID "{id}" timed out.')
                raise TimeoutError
    except (ValueError, json.decoder.JSONDecodeError, KeyError):
        logging.error(f'Reading response for instruction "{instruction_name}" with ID "{id}" failed.')
    finally:
        connection.settimeout(None)


def send_response(connection, instruction, status='ok', payload=None):
    response = {
        'id': instruction['id'],
        'instruction': instruction['instruction'],
        'status': status,
        'payload': calculate_payload_validation(payload)
    }

    logging.debug(f'Sending response.\n{json.dumps(response, indent=4)}')

    connection.sendall(json.dumps(response).encode("utf-8"))

    if payload is not None:
        connection.sendall(payload)


def read_payload(connection, instruction, retries=5):
    size = instruction['payload']['size']
    checksum = instruction['payload']['checksum']

    if size < 1:
        logging.warning(f'Expected payload but none got transmitted. An instruction might got skipped.')
        return

    tries = 0
    data = bytearray()

    while len(data) < size:
        data += connection.recv(size - len(data))

    while tries < retries and hashlib.sha1(data).hexdigest() != checksum:
        tries += 1
        logging.info(f'Payload checksum does not match. Retry #{tries}.')

        send_response(connection, instruction, status='invalid')
        data = connection.recv(size)

    if tries == retries:
        logging.error(f'Payload checksum are not match after retry #{tries} still. Aborting.')
        send_response(connection, instruction, status='failed')
    else:
        return data


def update_wifi(connection, instruction):
    data = read_payload(connection, instruction)

    if data is None:
        logging.error('Could not read Wi-Fi change payload.')
        return

    data = json.loads(data)

    logging.debug(f'Received Wi-Fi change details.\n{json.dumps(data, indent=4)}')

    # TODO add wifi via network manager dbus

    logging.info(f'Wi-Fi network "{data["ssid"]}" added.')
    send_response(connection, instruction)


def update(connection, instruction):
    data = read_payload(connection, instruction)

    if data is None:
        logging.error('Could not read update change payload.')
        return

    data = json.loads(data.decode('utf-8'))
    size = data["size"]
    package_size = data["packageSize"]
    buffer_size = package_size

    send_response(connection, instruction)

    with open(instruction['instruction'], 'w+b') as file:
        # noinspection PyShadowingBuiltins
        for part, hash in enumerate(data['hashes']):
            if part == (len(data['hashes']) - 1):
                buffer_size = size - part * buffer_size

            checksum = None

            while checksum != hash:
                print(part)

                buffer = bytearray()

                try:
                    connection.settimeout(30)

                    while buffer_size != len(buffer):
                        buffer += connection.recv(buffer_size - len(buffer))
                except socket.timeout:
                    logging.error(
                        f'Receiving update file timed out '
                        f'after {calculate_progress(size, part * package_size)}% has been received.'
                    )
                    return
                finally:
                    connection.settimeout(None)

                checksum = hashlib.sha1(buffer).hexdigest()

                if checksum == hash:
                    status = 'ok'
                    file.write(buffer)
                else:
                    status = 'invalid'
                    buffer.clear()

                    logging.warning(
                        f'Update file got corrupted at offset {part * package_size}, '
                        f'which is equals to {calculate_progress(size, part * package_size)}%. '
                        f'Retrying transmission from this offset.'
                    )

                send_response(connection, instruction, status=status, payload=json.dumps({
                    'type': data['type'],
                    'part': part,
                }).encode('utf-8'))

    # TODO implement update mechanism
    send_instruction(connection, 'update', json.dumps({
        'status': 'success'     # if status not 'success', the status will get shown in the app
    }).encode('utf-8'))


def bluetooth_loop():
    adapter = bus.get('org.bluez', '/org/bluez/hci0').Address
    s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    s.bind((adapter, 1))

    s.listen(1)

    while True:
        connection, address = s.accept()
        address = address[0]

        logging.info(f'Established connection with {address}.')

        while True:
            try:
                connection.settimeout(None)
                data = connection.recv(1024).decode('utf-8')

                try:
                    data = json.loads(data)
                except json.decoder.JSONDecodeError:
                    logging.warning(
                        f'Received invalid instructions "{data.strip()}". '
                        f'The android app might to be updated.'
                    )
                    continue

                logging.debug(f'Instruction received.\n{json.dumps(data, indent=4)}')
                try:
                    instruction = data['instruction']
                except KeyError:
                    logging.warning('Ignoring payload received as instruction. An instruction got skipped most likely.')
                    continue

                # noinspection PyShadowingBuiltins
                id = str(data['id'])

                if id[0] == 'w':
                    logging.warning(f'Received response for timed out instruction "{instruction}" with ID "{id}".')

                    payload_size = data['payload']['size']

                    if payload_size > 0:
                        connection.recv(payload_size)

                    continue

                if instruction == 'diagnostics':
                    logging.info('Diagnostics requested.')
                    send_response(connection, data, payload=json.dumps(diagnostics()).encode('utf-8'))
                elif instruction == 'history':
                    global samples

                    logging.info('Sample history requested.')
                    send_response(connection, data, payload=json.dumps(list(samples.values())).encode('utf-8'))
                elif instruction == 'sample':
                    sample(connection, data)
                elif instruction == 'wifi':
                    update_wifi(connection, data)
                elif instruction == 'update':
                    update(connection, data)
                else:
                    logging.warning('Unavailable instruction received.')
                    send_response(connection, data, status=f'Instruction "{instruction}" not supported')
            except UnicodeDecodeError:
                logging.warning('Received binary data as instruction.')
            except ConnectionError:
                logging.info(f'Connection with {address} lost.')
                break
            except TimeoutError:
                logging.info(f'Connection with {address} timed out.')
                break

        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='show debug information', action='store_true')
    parser.add_argument('--samples', default='samples.csv', help='samples file path used as mock-up history source')
    parser.add_argument('--preview', default='preview.jpg', help='preview image file path used as mock-up')
    parser.add_argument('--raw', default='raw.jpg', help='raw image file path used as mock-up')
    parser.add_argument(
        '--package-size',
        type=int,
        default=50,
        help='Kilobyte size in which a file gets split up for transmission. '
             'If less than one, files will not get split up.'
    )

    args = parser.parse_args()

    log_level = getattr(logging, 'DEBUG' if args.verbose else 'INFO', None)
    # noinspection SpellCheckingInspection
    logging.basicConfig(level=log_level, format='[%(levelname)s]\t(%(asctime)s)\t\t%(message)s')

    samples = sample_history(args.samples)

    while True:
        try:
            bluetooth_loop()
        except (Exception, json.decoder.JSONDecodeError, ValueError) as e:
            logging.error(f'Unexpected exception. Restarting.\n{e}\n\n{traceback.format_exc()}')
