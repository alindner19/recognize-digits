from json import dumps
from requests import post


class HomeAssistant:
    def __init__(self, url, token, port=8123):
        self._base_url = f"{url}:{port}/api/"
        self._headers = {
            "Authorization": f"Bearer {token}",
            "content-type": "application/json",
        }

    def update_state(self, entity, state, attributes):
        url = f"{self._base_url}states/{entity}"
        units = "pH" if entity.split('.')[1] == 'ph' else 'mV'
        attributes['unit_of_measurement'] = units
        data = {
            'entity_id': entity,
            'state': state,
            'attributes': attributes,
        }
        return post(url, headers=self._headers, data=dumps(data))

    def push_message(self, entity, message, title=None):
        url = f"{self._base_url}services/notify/{entity}"
        data = {'message': message}
        if title:
            data['title'] = title
        return post(url, headers=self._headers, data=dumps(data))
