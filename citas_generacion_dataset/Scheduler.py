import pandas as pd
from itertools import product
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import queue
import os
import base64
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.send']


class State:
    def __init__(self, current_occupation: dict, next_user: int, max_possible_score: int, slots_filled: int):
        self.current_occupation = current_occupation
        self.next_user = next_user
        self.max_possible_score = max_possible_score
        self.slots_filled = slots_filled

    def __lt__(self, other):
        return self.max_possible_score > other.max_possible_score


class Scheduler:
    def __init__(self, file_path, monday_date):
        self.data = pd.read_csv(file_path, dtype=str)
        self.data = self.data[self.data["¿Aceptas la recogida de datos y la participación en el muestreo?"] == "Si"]
        self.best_schedule = {}
        self.best_score = 0
        self.pq = queue.PriorityQueue()
        self.monday_date = monday_date

        # Generate all time slots (including previously skipped lunch hours)
        self.time_slots = [f"{day} {hour}:{minute:02d}" for day, hour, minute in product(
            ["Lunes", "Martes", "Miércoles", "Jueves"],
            range(9, 21),  # From 9:00 to 20:30
            [0, 30]
        )]

        self.total_users = len(self.data)
        self.total_slots = len(self.time_slots)
        self.start_time = datetime.now()

    def print_available_slots(self):
        print("\nFranjas horarias disponibles:")
        day_order = ["Lunes", "Martes", "Miércoles", "Jueves"]
        slots_by_day = {day: [] for day in day_order}

        for slot in self.time_slots:
            day, time = slot.split(" ", 1)
            slots_by_day[day].append(time)

        total_slots = 0
        for day in day_order:
            if slots_by_day[day]:
                times = sorted(slots_by_day[day], key=lambda x: (
                    int(x.split(':')[0]), int(x.split(':')[1])))
                print(f"{day}:")
                print("  " + ", ".join(times))
                total_slots += len(times)

        print(f"\nTotal de franjas disponibles: {total_slots}")
        print(f"Total de participantes: {self.total_users}\n")

    def parse_time(self, time_str):
        try:
            h, m = map(int, time_str.split(':'))
            return h * 60 + m
        except:
            return None

    def generate_slots(self, start_str, end_str):
        start = self.parse_time(start_str)
        end = self.parse_time(end_str)
        if start is None or end is None or start >= end:
            return []

        slots = []
        current = start
        while current < end:
            hour = current // 60
            minute = current % 60
            slots.append(f"{hour}:{minute:02d}")
            current += 30
        return slots

    def calculate_max_possible(self, slots_filled, next_user):
        remaining_slots = self.total_slots - slots_filled
        remaining_users = self.total_users - next_user
        return slots_filled + min(remaining_slots, remaining_users)

    def assign_places(self):
        initial_max = self.calculate_max_possible(0, 0)
        initial_state = State(
            current_occupation={},
            next_user=0,
            max_possible_score=initial_max,
            slots_filled=0
        )
        self.pq.put((-initial_state.max_possible_score, initial_state))

        while not self.pq.empty():
            _, current_state = self.pq.get()

            elapsed = (datetime.now() - self.start_time).total_seconds()
            print(f"\rCola: {self.pq.qsize():5} | Franjas: {current_state.slots_filled:3}/{
                  self.total_slots} | Mejor: {self.best_score:3} | Tiempo: {elapsed:.1f}s", end="", flush=True)

            if current_state.max_possible_score <= self.best_score:
                continue

            if current_state.next_user >= self.total_users or current_state.slots_filled == self.total_slots:
                if current_state.slots_filled > self.best_score:
                    self.best_score = current_state.slots_filled
                    self.best_schedule = current_state.current_occupation.copy()
                    print(f"\n¡Nuevo mejor resultado encontrado: {
                          self.best_score}/{self.total_slots}!")
                continue

            user = self.data.iloc[current_state.next_user]
            user_id = user["Dirección de correo electrónico"]
            scheduled = False

            for day in ["Lunes", "Martes", "Miércoles", "Jueves"]:
                if pd.isna(user[day]):
                    continue

                for time_range in user[day].split(","):
                    if scheduled:
                        break
                    time_range = time_range.strip()
                    if '-' not in time_range:
                        continue

                    start_str, end_str = time_range.split('-', 1)
                    start_str = start_str.strip()
                    end_str = end_str.strip()

                    for slot_time in self.generate_slots(start_str, end_str):
                        slot_key = f"{day} {slot_time}"

                        if slot_key not in current_state.current_occupation:
                            new_occupation = current_state.current_occupation.copy()
                            new_occupation[slot_key] = user_id

                            new_slots_filled = current_state.slots_filled + 1
                            new_max = self.calculate_max_possible(
                                new_slots_filled,
                                current_state.next_user + 1
                            )

                            new_state = State(
                                current_occupation=new_occupation,
                                next_user=current_state.next_user + 1,
                                max_possible_score=new_max,
                                slots_filled=new_slots_filled
                            )
                            self.pq.put(
                                (-new_state.max_possible_score, new_state))
                            scheduled = True
                            break

            new_max = self.calculate_max_possible(
                current_state.slots_filled,
                current_state.next_user + 1
            )
            new_state = State(
                current_occupation=current_state.current_occupation,
                next_user=current_state.next_user + 1,
                max_possible_score=new_max,
                slots_filled=current_state.slots_filled
            )
            self.pq.put((-new_state.max_possible_score, new_state))

    def print_final_schedule(self):
        print("\n\nHorario final (Franjas asignadas):")
        day_mapping = {"Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3}
        scheduled_users = set()

        # Sort slots chronologically
        sorted_slots = sorted(self.best_schedule.items(),
                              key=lambda x: (day_mapping[x[0].split()[0]],
                                             int(x[0].split()[
                                                 1].split(':')[0]),
                                             int(x[0].split()[1].split(':')[1])))

        for slot, user in sorted_slots:
            day, time = slot.split(" ", 1)
            appointment_date = self.monday_date + \
                timedelta(days=day_mapping[day])
            date_str = appointment_date.strftime("%d/%m/%Y")
            print(f"{day} {date_str} {time}: {user}")
            scheduled_users.add(user)

        print(f"\nResumen:")
        print(f"Franjas horarias asignadas: {len(self.best_schedule)}/{
              self.total_slots} ({len(self.best_schedule)/self.total_slots*100:.1f}%)")
        print(f"Participantes asignados: {len(
            scheduled_users)}/{self.total_users} ({len(scheduled_users)/self.total_users*100:.1f}%)")

        # Return scheduled users for email purposes
        return scheduled_users

    def send_emails(self, scheduled_users):
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=8080)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('gmail', 'v1', credentials=creds)
        day_mapping = {"Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3}

        # Send emails to scheduled users
        print("\nEnviando correos a participantes seleccionados...")
        for slot, user in self.best_schedule.items():
            day, time = slot.split(" ", 1)
            appointment_date = self.monday_date + \
                timedelta(days=day_mapping[day])
            date_str = appointment_date.strftime("%d/%m/%Y")

            subject = "Confirmación de cita para el estudio de FratosVR (con info adicional)"
            body = (
                f"Estimado/a participante,\n\n"
                f"Gracias por su interés en nuestro estudio.\n\n"
                f"Nos complace confirmar su cita para el {day} {date_str} "
                f"a las {time} en la Facultad de Informática de la UCM "
                f"(Calle del Prof. José García Santesmases, 9, Moncloa - Aravaca, 28040 Madrid).\n\n"
                f"Si tiene alguna pregunta, "
                f"no dude en ponerse en contacto con nosotros respondiendo a este correo.\n\n"
                f"Atentamente,\n"
                f"El equipo de FratosVR"
            )

            message = MIMEText(body)
            message["Subject"] = subject
            message["From"] = "alejba02@gmail.com"
            message["To"] = user

            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            try:
                send_message = service.users().messages().send(
                    userId="me", body={"raw": raw_message}).execute()
                print(f"\rCorreo enviado a {
                      user.ljust(40)}", end="", flush=True)
            except Exception as e:
                print(f"\nError al enviar correo a {user}: {str(e)}")

        # Send emails to non-selected users
        # print("\n\nEnviando correos a participantes no seleccionados...")
        # all_users = set(self.data["Dirección de correo electrónico"])
        # non_selected_users = all_users - scheduled_users

        # for user in non_selected_users:
        #     subject = "Agradecimiento por su interés en el estudio"
        #     body = (
        #         f"Estimado/a participante,\n\n"
        #         f"En primer lugar, queremos agradecerle sinceramente por su interés "
        #         f"en participar en nuestro estudio y por el tiempo que ha dedicado "
        #         f"a completar nuestro formulario.\n\n"
        #         f"Lamentamos informarle que, debido a limitaciones de tiempo y "
        #         f"a la gran cantidad de solicitudes recibidas, no hemos podido "
        #         f"asignarle una cita en esta ocasión.\n\n"
        #         f"Atentamente,\n"
        #         f"El equipo de FratosVR"
        #     )

        #     message = MIMEText(body)
        #     message["Subject"] = subject
        #     message["From"] = "alejba02@gmail.com"
        #     message["To"] = user

        #     raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        #     try:
        #         send_message = service.users().messages().send(
        #             userId="me", body={"raw": raw_message}).execute()
        #         print(f"\rCorreo enviado a {
        #               user.ljust(40)}", end="", flush=True)
        #     except Exception as e:
        #         print(f"\nError al enviar correo a {user}: {str(e)}")

        print("\n\nTodos los correos han sido enviados.")

    def run(self):
        print("Iniciando proceso de asignación de citas...")
        self.print_available_slots()
        print("Presione Ctrl+C para detener el proceso y usar el mejor horario encontrado")

        try:
            self.assign_places()
        except KeyboardInterrupt:
            print(
                "\n\n¡Proceso interrumpido! Usando el mejor horario encontrado hasta ahora...")

        print("\nResultados finales:")
        scheduled_users = self.print_final_schedule()

        if input("\n¿Enviar correos de confirmación? (s/n): ").lower() == 's':
            self.send_emails(scheduled_users)
        print("\nProceso completado con éxito.")


if __name__ == "__main__":
    monday_date = datetime(2025, 4, 7)
    scheduler = Scheduler("citas.csv", monday_date)
    scheduler.run()
