from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    """Vercel Python serverless entrypoint.

    This project is built as a Streamlit app, which is better hosted on
    Streamlit Community Cloud. This endpoint provides a successful Vercel
    deployment target and points users to the Streamlit app.
    """

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        message = (
            "ClimbAssist AI is a Streamlit application.\n\n"
            "Recommended hosting: Streamlit Community Cloud\n"
            "Main file: app_v2.py\n"
            "Run locally: streamlit run app_v2.py\n"
        )
        self.wfile.write(message.encode("utf-8"))
