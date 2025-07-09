# -*- coding: utf-8 -*-
import configparser
import os

import feapder


config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

start_page = int(config.get('screen', 'start_page'))
end_page = int(config.get('screen', 'end_page'))
thread_count = int(config.get('screen', 'thread_count'))
proxies = None
#proxies = config.get('screen', 'proxies')
folder_name = config.get('screen', 'folder_name')


def create_unique_file(base_filename):
    counter = 1
    if '.' in base_filename:
        ext = base_filename.split('.')[1]
        filename = base_filename.split('.')[0]
    else:
        ext = ''
        filename = base_filename

    while os.path.exists(filename):
        counter_str = f"{counter:02d}"
        filename = f"{base_filename}_{counter_str}.{ext}"
        counter += 1

    return filename + '.' + ext


class Spider(feapder.AirSpider):
    def start_requests(self):
        for page in range(start_page, end_page + 1):
            url = "https://samate.nist.gov/SARD/api/test-cases/search"
            params = {
                "page": str(page),
                "limit": "25"
            }

            yield feapder.Request(url, params=params)

    def download_midware(self, request):
        if proxies:
            request.proxies = {"https": proxies, "http": proxies}
        return request

    def parse(self, request, response):
        json_data = response.json
        test_case_list = json_data.get("testCases", [])
        for test_case in test_case_list:
            runs = test_case.get('sarif').get("runs", [])
            link = test_case.get("link")
            id = link.split('/')[-3]
            for run in runs:
                language = run['properties']['language']
                description = run['properties']['description']
                results = run.get("results", [])
                for result in results:
                    ruleId = result.get("ruleId", [])
                    locations = result.get("locations", [])
                    for location in locations:
                        uri = location['physicalLocation']['artifactLocation']['uri']
                        url = link + '/files/' + uri
                        yield feapder.Request(url, callback=self.parse_file, id=id, language=language, ruleId=ruleId, des=description)

    def parse_file(self, request, response):
        id = request.id
        language = request.language
        ruleId = request.ruleId
        description = request.des
        #print("description: ", type(description))
        #exit(0)
        file_name = str(id) + '_' + response.url.split('/')[-1]
        folder_language_name = os.path.join(folder_name, language)
        folder_language_name = os.path.join(folder_language_name, ruleId)
        #folder_language_name = os.path.join(folder_name, ruleId)

        if not os.path.exists(folder_language_name):
            os.makedirs(folder_language_name, exist_ok=True)

        file_name = create_unique_file(file_name)
        description_name = file_name.split('.')[0] + '.txt'
        with open(os.path.join(folder_language_name, file_name), 'wb') as f:
            f.write(response.content)
        with open(os.path.join(folder_language_name, description_name), 'w') as f:
            f.write(description)


if __name__ == "__main__":
    Spider(thread_count=thread_count).start()