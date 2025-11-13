package com.ui.pages.user;

import org.openqa.selenium.By;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class RegistrationPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By submitButton = By.cssSelector("[data-test='register-submit']");

    private final By firstNameErrorMessages = By.cssSelector("[data-test='first-name-error']");
    private final By lastNameErrorMessages = By.cssSelector("[data-test='last-name-error']");
    private final By emailErrorMessages = By.cssSelector("[data-test='email-error']");
    private final By passwordErrorMessages = By.cssSelector("[data-test='password-error']");
    private final By dobErrorMessages = By.cssSelector("[data-test='dob-error']");
    private final By streetErrorMessages = By.cssSelector("[data-test='street-error']");
    private final By cityErrorMessages = By.cssSelector("[data-test='city-error']");
    private final By stateErrorMessages = By.cssSelector("[data-test='state-error']");
    private final By countryErrorMessages = By.cssSelector("[data-test='country-error']");
    private final By phoneErrorMessages = By.cssSelector("[data-test='phone-error']");
    private final By postalCodeErrorMessages = By.cssSelector("[data-test='postal_code-error']");

    public RegistrationPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/auth/register");
    }

    public void submitForm() {
        driver.findElement(submitButton).click();
    }

    private String getErrorText(By locator) {
        try {
            return wait.until(ExpectedConditions.visibilityOfElementLocated(locator)).getText();
        } catch (TimeoutException e) {
            return "";
        }
    }

    public String getFirstNameErrorMessages() {
        return getErrorText(firstNameErrorMessages);
    }

    public String getLastNameErrorMessages() {
        return getErrorText(lastNameErrorMessages);
    }

    public String getEmailErrorMessages() {
        return getErrorText(emailErrorMessages);
    }

    public String getPasswordErrorMessages() {
        return getErrorText(passwordErrorMessages);
    }

    public String getDobErrorMessages() {
        return getErrorText(dobErrorMessages);
    }

    public String getStreetErrorMessages() {
        return getErrorText(streetErrorMessages);
    }

    public String getCityErrorMessages() {
        return getErrorText(cityErrorMessages);
    }

    public String getStateErrorMessages() {
        return getErrorText(stateErrorMessages);
    }

    public String getCountryErrorMessages() {
        return getErrorText(countryErrorMessages);
    }

    public String getPhoneErrorMessages() {
        return getErrorText(phoneErrorMessages);
    }

    public String getPostalCodeErrorMessages() {
        return getErrorText(postalCodeErrorMessages);
    }


}
