/*
 * uart_client.c
 *
 * Хост-программа для взаимодействия с Zephyr RTOS через UART (псевдотерминал).
 *
 * Программа:
 *   1. Открывает псевдотерминал /dev/pts/XXX, созданный Verilator
 *   2. Генерирует случайный пароль
 *   3. Побайтово отправляет пароль в UART
 *   4. Читает ответ от Zephyr (echo)
 *   5. Сравнивает отправленное с полученным
 *
 * Сборка:
 *   gcc -o uart_client uart_client.c
 *
 * Запуск:
 *   ./uart_client /dev/pts/XXX
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <time.h>

#define PASSWORD_LEN  8
#define TIMEOUT_SEC   30

/*
 * Настройка псевдотерминала в raw-режим.
 *
 * Raw-режим отключает канонический ввод (line buffering),
 * echo, и обработку управляющих символов ядром Linux.
 * Это необходимо, чтобы каждый байт передавался как есть,
 * без интерпретации со стороны line discipline.
 */
static int configure_pty(int fd)
{
    struct termios tty;

    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        return -1;
    }

    /* Raw-режим: отключить всю обработку на стороне ядра */
    cfmakeraw(&tty);

    /* Минимум 1 байт для read(), таймаут 10 * 0.1 = 1 секунда */
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 10;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        return -1;
    }

    return 0;
}

/*
 * Генерация случайного пароля из печатных ASCII-символов.
 * Используются символы a-z, A-Z, 0-9.
 */
static void generate_password(char *buf, int len)
{
    static const char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";

    srand((unsigned int)time(NULL));

    for (int i = 0; i < len; i++) {
        buf[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    buf[len] = '\0';
}

/*
 * Отправка строки побайтово в UART.
 *
 * Между байтами делается небольшая пауза (usleep),
 * чтобы симуляция Verilator успевала обработать каждый символ.
 * Verilator работает в ~1000-10000 раз медленнее реального чипа,
 * поэтому без паузы RX FIFO может переполниться.
 */
static int send_password(int fd, const char *password, int len)
{
    printf("[TX] Отправка пароля: %s\n", password);

    for (int i = 0; i < len; i++) {
        ssize_t written = write(fd, &password[i], 1);
        if (written != 1) {
            perror("write");
            return -1;
        }
        printf("[TX] Отправлен байт: '%c' (0x%02X)\n",
               password[i], (unsigned char)password[i]);

        /* Пауза 100 мс — даём Verilator время на обработку */
        usleep(100000);
    }

    return 0;
}

/*
 * Чтение ответа от Zephyr (echo).
 *
 * Функция читает побайтово с таймаутом.
 * Zephyr-программа возвращает каждый принятый символ обратно,
 * поэтому ожидается ровно len байт.
 */
static int receive_response(int fd, char *buf, int len)
{
    int received = 0;
    time_t start = time(NULL);

    printf("[RX] Ожидание ответа...\n");

    while (received < len) {
        /* Проверка общего таймаута */
        if (time(NULL) - start > TIMEOUT_SEC) {
            printf("[RX] Таймаут: получено %d из %d байт\n",
                   received, len);
            break;
        }

        ssize_t n = read(fd, &buf[received], 1);
        if (n == 1) {
            printf("[RX] Получен байт:  '%c' (0x%02X)\n",
                   buf[received], (unsigned char)buf[received]);
            received++;
        } else if (n == 0 || (n < 0 && errno == EAGAIN)) {
            /* Нет данных — ждём */
            usleep(50000);
        } else {
            perror("read");
            return -1;
        }
    }

    buf[received] = '\0';
    return received;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Использование: %s /dev/pts/XXX\n", argv[0]);
        return 1;
    }

    const char *pts_path = argv[1];

    /* Открытие псевдотерминала */
    printf("Открытие %s...\n", pts_path);
    int fd = open(pts_path, O_RDWR | O_NOCTTY);
    if (fd < 0) {
        perror("open");
        fprintf(stderr, "Не удалось открыть %s\n", pts_path);
        return 1;
    }

    /* Настройка raw-режима */
    if (configure_pty(fd) != 0) {
        close(fd);
        return 1;
    }

    /* Пауза для стабилизации соединения */
    printf("Подключено к %s\n", pts_path);
    printf("Ожидание готовности Zephyr (2 сек)...\n");
    sleep(2);

    /* Сброс входного буфера — убрать мусор от загрузки Zephyr */
    tcflush(fd, TCIOFLUSH);

    /* Генерация пароля */
    char password[PASSWORD_LEN + 1];
    generate_password(password, PASSWORD_LEN);

    printf("\n========================================\n");
    printf("  Сгенерирован пароль: %s\n", password);
    printf("========================================\n\n");

    /* Отправка пароля в Zephyr через UART */
    if (send_password(fd, password, PASSWORD_LEN) != 0) {
        close(fd);
        return 1;
    }

    printf("\n");

    /* Чтение ответа от Zephyr */
    char response[PASSWORD_LEN + 1];
    int received = receive_response(fd, response, PASSWORD_LEN);

    printf("\n========================================\n");
    printf("  Отправлено: %s\n", password);
    printf("  Получено:   %s\n", response);
    printf("========================================\n");

    /* Сравнение */
    if (received == PASSWORD_LEN && strcmp(password, response) == 0) {
        printf("  Результат:  СОВПАДЕНИЕ\n");
        printf("  Двустороннее взаимодействие подтверждено.\n");
    } else {
        printf("  Результат:  НЕСОВПАДЕНИЕ\n");
        printf("  Получено %d из %d байт.\n", received, PASSWORD_LEN);
    }
    printf("========================================\n");

    close(fd);
    return 0;
}
