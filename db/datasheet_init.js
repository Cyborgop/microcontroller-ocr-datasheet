use microcontrollers

db.datasheets.insertMany([
  {
    part_number: "LPC1768",
    manufacturer: "NXP Semiconductors",
    core: "ARM Cortex-M3",
    flash: "512KB",
    ram: "64KB",
    max_clock: "100MHz",
    datasheet_url: "https://www.nxp.com/docs/en/data-sheet/LPC1768_66.pdf",
    features: [
      "32-bit MCU",
      "Ethernet, USB, CAN",
      "Multiple UARTs, ADC, DAC",
      "Timers, PWM, I2C, SPI"
    ]
  },
  {
    part_number: "8051",
    manufacturer: "Intel (original), many vendors",
    core: "8051 (MCS-51)",
    flash: "Varies (typically 4KB–64KB)",
    ram: "128B–4KB",
    max_clock: "12–40MHz",
    datasheet_url: "https://www.ti.com/lit/ds/symlink/mcs51.pdf",
    features: [
      "8-bit MCU",
      "Harvard architecture",
      "UART, Timers, Interrupts",
      "Widely used in education and industry"
    ]
  },
  {
    part_number: "ATmega328P",
    manufacturer: "Microchip (Atmel)",
    core: "AVR 8-bit",
    flash: "32KB",
    ram: "2KB",
    max_clock: "20MHz",
    datasheet_url: "https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-7810-Automotive-Microcontrollers-ATmega328P_Datasheet.pdf",
    features: [
      "8-bit AVR RISC architecture",
      "23 I/O lines",
      "10-bit ADC",
      "6 PWM channels",
      "Used in Arduino Uno"
    ]
  }
])
