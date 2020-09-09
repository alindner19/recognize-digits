import os
from argparse import ArgumentParser
from pathlib import Path
from homeassistant import HomeAssistant
from recognize_digits.recognize_digits import RecognizeDigits, ReadDigitsImage
from recognize_digits.exceptions import DataException
from secrets import HOME_URL, HA_TOKEN

ALARM_TIMEOUT = 2


def yaml_path():
    if os.name == "nt":
        return Path('s:\\') / 'python' / 'chemtrol'
    return Path("/volume1/data/")


class ChemtrolDigits(RecognizeDigits):
    @property
    def alarm_on(self):
        return self._attrs.get('alarm', False)

    @property
    def as_int(self):
        data = self.read()
        if self.alarm_on:
            print(self._attrs.get('name'), 'alarm is on', data)
            return ''

        if data.isdigit():
            value = int(data)
            if value < 100:
                print('uncaught timeout error', value)
                return ''

            return value
        return ''

    @property
    def as_float(self):
        try:
            value = float(self.read())
        except ValueError:
            value = ''
        if self.alarm_on:
            print(self._attrs.get('name'), 'alarm is on', value)
            return ''

        if isinstance(value, float) and (value < 5 or value >= 10):
            print('uncaught timeout error', value)
            return ''
        return value


class ChemtrolImage(ReadDigitsImage):
    def digits(self):
        image = self.conditioned_image()
        recd = {}
        for dgt in self.options['digits']:
            recd[dgt['name']] = ChemtrolDigits(image, debug=self.debug, **dgt)

        return recd

    def get_digits(self, count=0):
        recd = self.digits()
        count = 0
        while not recd and count < ALARM_TIMEOUT:
            sleep(0.5)
            count += 1
            recd = self.digits(count)

        if count == ALARM_TIMEOUT:
            return None

        return recd['orp'], recd['ph']

    def push(self):
        orp, ph = self.get_digits()
        ha = HomeAssistant(HOME_URL, HA_TOKEN)
        state = orp.as_int
        attributes = orp.attributes
        ha.update_state("sensor.orp", state, attributes)

        state = ph.as_float
        attributes = ph.attributes
        ha.update_state("sensor.ph", state, attributes)


def main(images, push, to_file, debug):
    path = yaml_path() / 'chemtrol.yaml'
    chemtrol = ChemtrolImage(path, debug=debug)
    if to_file:
        chemtrol.save()
        return

    if push:
        chemtrol.push()
        return

    if images:
        chemtrol.debug_images()
        return

    orp, ph = chemtrol.get_digits()

    try:
        print('ORP: {} ({}%)'.format(orp.as_int, orp.worst_margin))
    except DataException as reason:
        print(reason)
    for attr, value in orp.attributes.items():
        print('  {}: {}'.format(attr, value))
    try:
        print('PH: {} ({}%)'.format(ph.as_float, ph.worst_margin))
    except DataException as reason:
        print(reason)
    for attr, value in ph.attributes.items():
        print('  {}: {}'.format(attr, value))


if __name__ == "__main__":
    parser = ArgumentParser(description="Chemtrol camera read access")
    parser.add_argument(
        '-i', '--images',
        action='store_true',
        default=False,
        help="Display images for crop and warp points."
    )
    parser.add_argument(
        '-f', '--to_file',
        action='store_true',
        default=False,
        help="Save results to file."
    )
    parser.add_argument(
        '-p', '--push',
        action='store_true',
        default=False,
        help="Push result to home assistant."
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        default=False,
        help="Debug mode."
    )
    args = parser.parse_args()
    main(args.images, args.push, args.to_file, args.debug)

