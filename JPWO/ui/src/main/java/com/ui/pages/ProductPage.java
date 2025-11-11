package com.ui.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.NoSuchElementException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class ProductPage {

    private WebDriver driver;
    private WebDriverWait wait;

    private By addToCartButton = By.cssSelector("[data-test='add-to-cart']");
    private By cartBadge = By.cssSelector("[data-test='cart-quantity']");
    private By cartButton = By.cssSelector("[data-test='nav-cart']");

    public ProductPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void open(String productId) {
        driver.get("http://localhost:4200/product/" + productId);
    }

    public void addToCart() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(addToCartButton));
        button.click();
    }

    public void openCartPage() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(cartButton));
        button.click();
    }

    public int getCartCount() {
        try {
            String countText = wait.until(
                    ExpectedConditions.visibilityOfElementLocated(cartBadge)
            ).getText();
            return Integer.parseInt(countText);
        } catch (NoSuchElementException e) {
            return 0;
        }
    }
}
