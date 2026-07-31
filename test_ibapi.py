from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import time

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def nextValidId(self, orderId):
        print("✅ Connected. nextValidId:", orderId)
        self.reqManagedAccts()

    def managedAccounts(self, accountsList):
        print("✅ Accounts:", accountsList)
        self.disconnect()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"❌ Error {errorCode}, reqId {reqId}: {errorString}")

app = TestApp()
print("Connecting to 127.0.0.1:4002 ...")
app.connect("127.0.0.1", 4002, clientId=99)

app.run()