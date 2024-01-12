
import re
import requests
from bs4 import BeautifulSoup


class uniqueCourse:
    def __init__(self, assignments, courses, modules, course_details, course_id):
        self.assignments = assignments
        self.courses = courses
        self.modules = modules
        self.course_details = course_details
        self.course_id = course_id
        # Instantiate CanvasAPI here
        self.canvas_api = CanvasAPI(
            '1017~zvygZeqjowJTFeOvTUhByFSAwdRhs0Ccng5pmtCmq3iP5XW3RtDdjg2g6scuPoxj')
        self.ret = ""

    def assignmentRep(self):
        if self.assignments:
            for assignment in self.assignments:
                assignment_details = self.canvas_api.get_assignment_details(
                    self.course_id, assignment['id'])
                if assignment_details:
                    assignment_details = {
                        'name': assignment['name'],
                        'due_date': assignment['due_at'],
                        'description': assignment['description'],
                        'submission_types': assignment['submission_types'],
                        'grading_type': assignment['grading_type'],
                        'points_possible': assignment['points_possible'],
                    }
                    for key, value in assignment_details.items():
                        if value != None:
                            self.ret += str(key) + " " + str(value) + "\n"
                    self.ret += ("\n")

    def moduleRep(self):
        modules = self.canvas_api.get_modules(self.course_id)
        if modules:
            for module in modules:
                module_id = module['id']
                module_items = self.canvas_api.get_module_items(
                    self.course_id, module_id)
                if module_items:
                    self.ret += (f"Module Name: {module['name']}\n")

    def announcemnt_rep(self):
        announcements = self.canvas_api.get_announcements(self.course_id)
        if announcements:
            for announcement in announcements:

                html_content = announcement["message"]

                soup = BeautifulSoup(html_content, 'html.parser')

                # Extracting text content while removing <p> tags
                text_content = ''.join(
                    soup.find_all(text=True, recursive=False))
                self.ret += announcement["title"] + "\n"
                self.ret += text_content + "\n"

    def grades(self):
        grades = self.canvas_api.get_grades(self.course_id)
        return str(grades)

    def stringRep(self):
        self.assignmentRep()
        self.moduleRep()
        self.announcemnt_rep()
        return self.ret


class CanvasAPI:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = "https://utexas.instructure.com/api/v1/"

    def _make_request(self, endpoint):
        headers = {
            'Authorization': f'Bearer {self.access_token}',
        }
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            # Handle errors here

            return None

    def get_assignments(self, course_id):
        return self._make_request(f"courses/{course_id}/assignments")

    def get_grades(self, course_id):
        return self._make_request(f"courses/{course_id}/grades")

    def get_announcements(self, course_id):
        return self._make_request(f"courses/{course_id}/discussion_topics?only_announcements=true")

    def get_modules(self, course_id):
        return self._make_request(f"courses/{course_id}/modules")

    def get_courses(self):
        return self._make_request("courses")

    def get_course_details(self, course_id):
        return self._make_request(f"courses/{course_id}")

    def get_assignment_details(self, course_id, assignment_id):
        return self._make_request(f"courses/{course_id}/assignments/{assignment_id}")

    def get_module_items(self, course_id, module_id):
        return self._make_request(f"courses/{course_id}/modules/{module_id}/items")
