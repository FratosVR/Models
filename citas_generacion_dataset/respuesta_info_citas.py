import base64
import pickle
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.text import MIMEText

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
DESPACHO_INFO = "Despacho Único - Calle Principal 123, Madrid, España"


def get_gmail_service():
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

    return build('gmail', 'v1', credentials=creds)


def find_confirmation_emails(service):
    query = "in:inbox Confirmación de cita para el estudio de FratosVR (con info adicional)"
    return service.users().messages().list(
        userId='me',
        q=query,
        maxResults=10
    ).execute().get('messages', [])


def parse_email_content(service, message_id):
    msg = service.users().messages().get(
        userId='me',
        id=message_id,
        format='full'
    ).execute()

    headers = {h['name']: h['value'] for h in msg['payload']['headers']}
    return {
        'id': message_id,
        'from': headers['From'],
        'subject': headers['Subject']
    }


def create_reply(original_email):
    reply_body = f"""
Estimado/a/e participante,

Gracias por confirmar su cita. En caso de haber declinado la cita, ignore este correo.

Le informamos que su cita tendrá lugar en el despacho:
{DESPACHO_INFO}

Atentamente,
El equipo de FratosVR
    """

    message = MIMEText(reply_body)
    message['To'] = original_email['from']
    message['Subject'] = 'Re: ' + original_email['subject']
    message['In-Reply-To'] = original_email['id']
    message['References'] = original_email['id']

    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}


def process_emails(service):
    messages = find_confirmation_emails(service)

    for msg in messages:
        email = parse_email_content(service, msg['id'])
        print(f"Procesando confirmación de {email['from']}")

        reply = create_reply(email)
        service.users().messages().send(
            userId='me',
            body=reply
        ).execute()

        print(f"Respuesta enviada con ubicación única: {DESPACHO_INFO}")
        print("---")


if __name__ == '__main__':
    service = get_gmail_service()
    print("Buscando confirmaciones de citas...")
    process_emails(service)
    print("Proceso completado. Todas las respuestas enviadas.")
