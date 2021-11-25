import bluetooth

class BL:
    def __init__(self):
        self.__RFCOMM_UUID = "00001101-0000-1000-8000-00805f9b34fb"

        self.server_sock = None
        self.port = 1
        self.client_sock = None
        self.client_info = ''

        self.state = False
        self.close = False

    def start_server(self):
        server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        server_sock.bind(("", bluetooth.PORT_ANY))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        bluetooth.advertise_service(server_sock, "sopoware", service_id=self.__RFCOMM_UUID,
                                    service_classes=[self.__RFCOMM_UUID, bluetooth.SERIAL_PORT_CLASS],
                                    profiles=[bluetooth.SERIAL_PORT_PROFILE])

        self.server_sock = server_sock
        self.port = port

    def wait_client(self):
        print("Waiting for connection on RFCOMM channel", self.port)

        client_sock, client_info = self.server_sock.accept()
        print("Accepted connection from", client_info)

        self.client_sock = client_sock
        self.client_info = client_info

    def cleanup(self):
        print('Closing bluetooth socket')
        if not self.client_sock == None:
            self.client_sock.close()
        if not self.server_sock == None:
            self.server_sock.close()

    def cycle(self):
        self.start_server()

        while not self.close:
            self.wait_client()
            self.state = True
            try:
                while True and (not self.close or not self.state):
                    data = self.client_sock.recv(1024)
                    if not data:
                        self.state = False
                        break
                    if data:
                        self.client_sock.send(data)
                        data = data.decode('utf-8')
                        if data == '1':
                            self.state = True
                        elif data == '0':
                            self.state = False
                        elif data == '-1':
                            self.state = False
                            self.close = True
                            break
            except OSError:
                print('Client Disconnected')
                self.state = False
                pass