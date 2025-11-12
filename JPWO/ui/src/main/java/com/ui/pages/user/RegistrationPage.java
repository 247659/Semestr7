package com.ui.pages.user;

import org.openqa.selenium.By;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class RegistrationPage {

    private WebDriver driver;
    private WebDriverWait wait;

    private By submitButton = By.cssSelector("[data-test='register-submit']");

    private By firstNameErrorMessages = By.cssSelector("[data-test='first-name-error']");
    private By lastNameErrorMessages = By.cssSelector("[data-test='last-name-error']");
    private By emailErrorMessages = By.cssSelector("[data-test='email-error']");
    private By passwordErrorMessages = By.cssSelector("[data-test='password-error']");
    private By dobErrorMessages = By.cssSelector("[data-test='dob-error']");
    private By streetErrorMessages = By.cssSelector("[data-test='street-error']");
    private By cityErrorMessages = By.cssSelector("[data-test='city-error']");
    private By stateErrorMessages = By.cssSelector("[data-test='state-error']");
    private By countryErrorMessages = By.cssSelector("[data-test='country-error']");
    private By phoneErrorMessages = By.cssSelector("[data-test='phone-error']");
    private By postalCodeErrorMessages = By.cssSelector("[data-test='postal_code-error']");

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
