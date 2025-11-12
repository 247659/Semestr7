package com.ui.tests;

import com.ui.pages.RegistrationPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ValidateRegistrationTest {

    private WebDriver driver;
    private RegistrationPage registrationPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        registrationPage = new RegistrationPage(driver);
        registrationPage.open();
    }

    @Test
    void shouldShowErrorsForAllEmptyFields() {
        registrationPage.submitForm();

        assertEquals("First name is required", registrationPage.getFirstNameErrorMessages());
        assertEquals("Last name is required", registrationPage.getLastNameErrorMessages());
        assertEquals("Email is required", registrationPage.getEmailErrorMessages());
        assertTrue(registrationPage.getPasswordErrorMessages().contains("Password is required"));
        assertTrue(registrationPage.getDobErrorMessages().contains("Date of Birth is required"));
        assertEquals("Street is required", registrationPage.getStreetErrorMessages());
        assertEquals("City is required", registrationPage.getCityErrorMessages());
        assertEquals("State is required", registrationPage.getStateErrorMessages());
        assertEquals("Country is required", registrationPage.getCountryErrorMessages());
        assertEquals("Phone is required.", registrationPage.getPhoneErrorMessages());
        assertEquals("Postcode is required", registrationPage.getPostalCodeErrorMessages());
    }

    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
