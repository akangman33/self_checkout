import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont, QPalette, QBrush, QPixmap
from PyQt5.QtCore import *
import qdarkstyle
from practice_signin import SignInWidget
from practice_signup import SignUpWidget
from practice_checkout import CheckOutWidget
import sip


class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.layout = QHBoxLayout()
        self.widget = SignInWidget()
        self.resize(1920, 1080)
        self.setWindowTitle("歡迎使用自動結帳系統")
        self.setCentralWidget(self.widget)
        bar = self.menuBar()
        self.Menu = bar.addMenu("選單")
        self.signUpAction = QAction("會員註冊", self)
        self.signInAction = QAction("會員登入", self)
        self.quitSignInAction = QAction("會員登出", self)
        self.checkoutAction = QAction("商品結帳", self)
        self.quitAction = QAction("退出", self)
        self.Menu.addAction(self.signUpAction)
        self.Menu.addAction(self.signInAction)
        self.Menu.addAction(self.quitSignInAction)
        self.Menu.addAction(self.checkoutAction)
        self.Menu.addAction(self.quitAction)
        self.signUpAction.setEnabled(True)
        self.checkoutAction.setEnabled(False)
        self.signInAction.setEnabled(False)
        self.quitSignInAction.setEnabled(False)
        self.widget.is_checkout_signal.connect(self.checkout)
        self.Menu.triggered[QAction].connect(self.menuTriggered)
        self.palette1 = QPalette()
        self.palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('./background/open_neon_sign_4k-1920x1080.jpg')))
        self.setPalette(self.palette1)

    def checkout(self, result):
        sip.delete(self.widget)
        self.widget = CheckOutWidget(result)
        self.setCentralWidget(self.widget)
        self.signUpAction.setEnabled(True)
        self.checkoutAction.setEnabled(True)
        self.signInAction.setEnabled(False)
        self.quitSignInAction.setEnabled(True)

    def menuTriggered(self, q):
        if(q.text()=="商品結帳"):
            sip.delete(self.widget)
            self.widget = CheckOutWidget()
            self.setCentralWidget(self.widget)
            self.is_model_signal.connect(self.checkout)
            self.signUpAction.setEnabled(True)
            self.checkoutAction.setEnabled(True)
            self.signInAction.setEnabled(False)
            self.quitSignInAction.setEnabled(True)
        if (q.text() == "會員註冊"):
            sip.delete(self.widget)
            self.widget = SignUpWidget()
            self.setCentralWidget(self.widget)
            self.signUpAction.setEnabled(False)
            self.checkoutAction.setEnabled(False)
            self.signInAction.setEnabled(True)
            self.quitSignInAction.setEnabled(False)
        if (q.text() == "會員登出"):
            sip.delete(self.widget)
            self.widget = SignInWidget()
            self.setCentralWidget(self.widget)
            self.widget.is_checkout_signal.connect(self.checkout)
            self.signUpAction.setEnabled(True)
            self.checkoutAction.setEnabled(False)
            self.signInAction.setEnabled(False)
            self.quitSignInAction.setEnabled(False)
        if (q.text() == "會員登入"):
            sip.delete(self.widget)
            self.widget = SignInWidget()
            self.setCentralWidget(self.widget)
            self.widget.is_checkout_signal.connect(self.checkout)
            self.signUpAction.setEnabled(True)
            self.checkoutAction.setEnabled(False)
            self.signInAction.setEnabled(False)
            self.quitSignInAction.setEnabled(False)
        if (q.text() == "退出"):
            qApp = QApplication.instance()
            qApp.quit()
        return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow.jpg"))
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainWindow = Main()
    mainWindow.show()
    sys.exit(app.exec_())
