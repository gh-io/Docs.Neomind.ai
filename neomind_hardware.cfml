<neomind_hardware>
    <name>Neomind Board v1</name>
    <usb_port>
        <type>USB-B</type>
        <supports>power, serial, firmware_upload</supports>
    </usb_port>

    <microcontroller>
        <chip>ATmega328P</chip>
        <flash>32KB</flash>
        <sram>2KB</sram>
        <eeprom>1KB</eeprom>
        <clock>16MHz</clock>
        <features>
            <interrupt_engine>true</interrupt_engine>
            <pwm_engine>true</pwm_engine>
            <adc_10bit>true</adc_10bit>
        </features>
    </microcontroller>

    <digital_pins count="14" starts_at="0">
        <pwm>[3,5,6,9,10,11]</pwm>
        <serial rx="0" tx="1"/>
    </digital_pins>

    <analog_pins count="6">
        <pins>A0,A1,A2,A3,A4,A5</pins>
        <adc_resolution>10bit</adc_resolution>
    </analog_pins>

    <power>
        <pins>GND,5V,3.3V,VIN</pins>
    </power>

    <reset_button>true</reset_button>
</neomind_hardware>
