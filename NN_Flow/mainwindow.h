#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <netlayer.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	bool stopTraning = false;
	bool calculated = false;
	convLayer *layer1_X28_Y28_NX14_NY14;
	convLayer *layer2_X14_Y14_NX5_NY5;
	convLayer *layer3_X5_Y5_NX5_NY1;

	traningSampleImg *inputDataBase;

	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

private slots:
	void on_pushButton_pressed();

	void on_spinBox_valueChanged(int arg1);

	void on_spinBox_2_valueChanged(int arg1);

	void on_spinBox_3_valueChanged(int arg1);

	void on_spinBox_4_valueChanged(int arg1);

	void show_L2_out(void);

    void show_L3_out(void);

    void on_spinBox_5_valueChanged(int arg1);

	void on_spinBox_valueChanged_inputImage(int arg1);

    void refresh_ui(void);


    // запись данных на вход сети
	void writeDataToLayer(int numImg);

	void on_pushButton_2_pressed();

	int grayToColorTemperature(int color);

private:
	Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
