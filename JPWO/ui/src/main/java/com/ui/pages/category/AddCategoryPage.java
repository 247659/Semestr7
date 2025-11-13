package com.ui.pages.category;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class AddCategoryPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By categoryNameField = By.cssSelector("[data-test='name']");
    private final By slugField = By.cssSelector("[data-test='slug']");
    private final By successMessage = By.className("alert-success");

    private final By submitButton = By.cssSelector("[data-test='category-submit']");

    public AddCategoryPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/admin/categories/add");
    }

    public void enterCategoryName(String name) {
        WebElement productName = wait.until(ExpectedConditions.visibilityOfElementLocated(categoryNameField));
        productName.clear();
        productName.sendKeys(name);
    }

    public void enterSlug(String slug) {
        WebElement element = driver.findElement(slugField);
        element.clear();
        element.sendKeys(slug);
    }


    public void submitForm() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(submitButton));
        button.click();

        wait.until(ExpectedConditions.visibilityOfElementLocated(successMessage));
    }

    public void fillForm(String name, String slug) {
        enterCategoryName(name);
        enterSlug(slug);
        submitForm();
    }
}
